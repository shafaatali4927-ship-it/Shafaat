import os
import asyncio
import tempfile
import subprocess
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

# ─── Whisper transcribe ───────────────────────────────────────────────────────
async def transcribe_audio(file_path: str, language: str = "ur") -> dict:
    import whisper
    model = whisper.load_model("medium")

    result = model.transcribe(file_path, language=language, verbose=False)

    # SRT
    def fmt_time(s):
        h, m = int(s // 3600), int((s % 3600) // 60)
        sec, ms = int(s % 60), int((s % 1) * 1000)
        return f"{h:02}:{m:02}:{sec:02},{ms:03}"

    srt_lines = []
    for i, seg in enumerate(result["segments"], 1):
        srt_lines.append(
            f"{i}\n{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}\n{seg['text'].strip()}\n"
        )

    return {
        "srt": "\n".join(srt_lines),
        "txt": result["text"].strip(),
        "language": result.get("language", "unknown"),
        "segments": len(result["segments"]),
    }

# ─── Video enhance ────────────────────────────────────────────────────────────
async def enhance_video(input_path: str, output_path: str):
    import torch, cv2
    from gfpgan import GFPGANer
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import ffmpeg, shutil

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4, model_path="models/RealESRGAN_x4plus.pth",
        model=rrdb, tile=256, tile_pad=10, pre_pad=0,
        half=(device == "cuda")
    )
    restorer = GFPGANer(
        model_path="models/GFPGANv1.4.pth", upscale=2,
        arch="clean", channel_multiplier=2, bg_upsampler=upsampler
    )

    frames_dir = tempfile.mkdtemp()
    out_dir = tempfile.mkdtemp()

    probe = ffmpeg.probe(input_path)
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    fps = eval(video_stream["r_frame_rate"])

    ffmpeg.input(input_path).output(
        f"{frames_dir}/frame_%06d.png", q=2
    ).overwrite_output().run(quiet=True)

    for fname in sorted(os.listdir(frames_dir)):
        img = cv2.imread(os.path.join(frames_dir, fname), cv2.IMREAD_COLOR)
        try:
            _, _, enhanced = restorer.enhance(
                img, has_aligned=False, only_center_face=False,
                paste_back=True, weight=0.5
            )
        except Exception:
            enhanced = img
        cv2.imwrite(os.path.join(out_dir, fname), enhanced)

    tmp_video = output_path + "_tmp.mp4"
    ffmpeg.input(f"{out_dir}/frame_%06d.png", framerate=fps).output(
        tmp_video, vcodec="libx264", pix_fmt="yuv420p", crf=18
    ).overwrite_output().run(quiet=True)

    audio_file = output_path + "_audio.aac"
    has_audio = True
    try:
        ffmpeg.input(input_path).output(audio_file, acodec="copy").overwrite_output().run(quiet=True)
    except Exception:
        has_audio = False

    if has_audio:
        ffmpeg.output(
            ffmpeg.input(tmp_video), ffmpeg.input(audio_file),
            output_path, vcodec="copy", acodec="aac"
        ).overwrite_output().run(quiet=True)
    else:
        shutil.copy(tmp_video, output_path)

    shutil.rmtree(frames_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    for f in [tmp_video, audio_file]:
        if os.path.exists(f):
            os.remove(f)

# ─── Handlers ─────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎙️ *Islamic Beacon — AI Bot*\n\n"
        "Kya bhej sakte ho:\n"
        "• 🎵 Audio file → SRT + TXT milega\n"
        "• 🎬 Video file → Enhanced video milegi\n\n"
        "Bas file bhejo, baki main karunga! ✅",
        parse_mode="Markdown"
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Audio mil gayi! Transcription shuru ho rahi hai...")

    audio = update.message.audio or update.message.voice or update.message.document
    file = await audio.get_file()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    # Language keyboard
    keyboard = [
        [InlineKeyboardButton("🇵🇰 Urdu", callback_data=f"lang_ur_{tmp_path}"),
         InlineKeyboardButton("🇸🇦 Arabic", callback_data=f"lang_ar_{tmp_path}")],
        [InlineKeyboardButton("🇬🇧 English", callback_data=f"lang_en_{tmp_path}"),
         InlineKeyboardButton("🔍 Auto Detect", callback_data=f"lang_auto_{tmp_path}")]
    ]
    await msg.edit_text(
        "🌐 Kaunsi language hai audio mein?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🎬 Video mil gayi! Enhancement shuru ho rahi hai... (5-15 min)")

    video = update.message.video or update.message.document
    file = await video.get_file()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        input_path = tmp.name

    output_path = input_path.replace(".mp4", "_enhanced.mp4")

    try:
        await enhance_video(input_path, output_path)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        await msg.edit_text(f"✅ Enhancement done! Size: {size_mb:.1f} MB — Bhej raha hun...")
        with open(output_path, "rb") as f:
            await update.message.reply_video(f, caption="🎬 AI Enhanced Video — Islamic Beacon ✨")
    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)}")
    finally:
        for p in [input_path, output_path]:
            if os.path.exists(p):
                os.remove(p)

async def handle_language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    parts = query.data.split("_", 2)
    lang = parts[1]
    tmp_path = parts[2]

    lang_display = {"ur": "Urdu 🇵🇰", "ar": "Arabic 🇸🇦", "en": "English 🇬🇧", "auto": "Auto 🔍"}.get(lang, lang)
    await query.edit_message_text(f"⏳ Transcribing ({lang_display})... thoda sabr karo!")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: asyncio.run(transcribe_audio(tmp_path, None if lang == "auto" else lang))
        )

        # SRT file
        srt_path = tmp_path.replace(".mp3", ".srt")
        txt_path = tmp_path.replace(".mp3", ".txt")

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(result["srt"])
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["txt"])

        await query.edit_message_text(
            f"✅ Done!\n🌐 Language: {result['language']}\n📝 Segments: {result['segments']}"
        )

        with open(srt_path, "rb") as f:
            await query.message.reply_document(f, filename="subtitles.srt", caption="📄 SRT File")
        with open(txt_path, "rb") as f:
            await query.message.reply_document(f, filename="transcript.txt", caption="📄 Plain Text")

    except Exception as e:
        await query.edit_message_text(f"❌ Error: {str(e)}")
    finally:
        for p in [tmp_path, srt_path, txt_path]:
            if os.path.exists(p):
                os.remove(p)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.AUDIO | filters.VOICE, handle_audio))
    app.add_handler(MessageHandler(
        filters.VIDEO | (filters.Document.MimeType("video/mp4")), handle_video
    ))
    app.add_handler(CallbackQueryHandler(handle_language_callback, pattern="^lang_"))

    print("🤖 Bot chal raha hai...")
    app.run_polling()

if __name__ == "__main__":
    main()
