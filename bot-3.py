import os
import sys
import asyncio
import tempfile
import shutil
import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

# ─── Torchvision Compatibility Patch ─────────────────────────────────────────
try:
    import torchvision.transforms.functional_tensor
except ModuleNotFoundError:
    import torchvision.transforms.functional as _F
    sys.modules["torchvision.transforms.functional_tensor"] = _F

# ─── Whisper Transcribe ───────────────────────────────────────────────────────
def do_transcribe(file_path, language):
    import whisper

    print("Loading whisper model...")
    model = whisper.load_model("small")

    opts = {}
    if language and language != "auto":
        opts["language"] = language

    result = model.transcribe(file_path, verbose=False, **opts)

    def fmt(s):
        h, m = int(s // 3600), int((s % 3600) // 60)
        sec, ms = int(s % 60), int((s % 1) * 1000)
        return f"{h:02}:{m:02}:{sec:02},{ms:03}"

    srt_lines = []
    for i, seg in enumerate(result["segments"], 1):
        srt_lines.append(
            f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{seg['text'].strip()}\n"
        )

    return {
        "srt": "\n".join(srt_lines),
        "txt": result["text"].strip(),
        "language": result.get("language", "?"),
        "segments": len(result["segments"]),
    }

# ─── Video Enhance ────────────────────────────────────────────────────────────
def do_enhance(input_path, output_path):
    import torch, cv2
    from gfpgan import GFPGANer
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import ffmpeg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Enhance device: {device}")

    rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path="models/RealESRGAN_x4plus.pth",
        model=rrdb, tile=256, tile_pad=10, pre_pad=0,
        half=(device == "cuda")
    )
    restorer = GFPGANer(
        model_path="models/GFPGANv1.4.pth",
        upscale=2, arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler
    )

    frames_dir = tempfile.mkdtemp()
    out_dir = tempfile.mkdtemp()
    tmp_video = None
    audio_file = None

    try:
        probe = ffmpeg.probe(input_path)
        video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        fps = eval(video_stream["r_frame_rate"])

        ffmpeg.input(input_path).output(
            f"{frames_dir}/frame_%06d.png", q=2
        ).overwrite_output().run(quiet=True)

        frame_files = sorted(os.listdir(frames_dir))
        print(f"Enhancing {len(frame_files)} frames...")

        for fname in frame_files:
            img = cv2.imread(os.path.join(frames_dir, fname), cv2.IMREAD_COLOR)
            try:
                _, _, enhanced = restorer.enhance(
                    img, has_aligned=False,
                    only_center_face=False,
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
            ffmpeg.input(input_path).output(
                audio_file, acodec="copy"
            ).overwrite_output().run(quiet=True)
        except Exception:
            has_audio = False

        if has_audio:
            ffmpeg.output(
                ffmpeg.input(tmp_video),
                ffmpeg.input(audio_file),
                output_path, vcodec="copy", acodec="aac"
            ).overwrite_output().run(quiet=True)
        else:
            shutil.copy(tmp_video, output_path)

    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        for f in [tmp_video, audio_file]:
            if f and os.path.exists(f):
                os.remove(f)

# ─── Telegram Handlers ────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎙️ *Islamic Beacon AI Bot*\n\n"
        "📤 Kya bhej sakte ho:\n"
        "• 🎵 Audio → SRT + TXT milega\n"
        "• 🎬 Video → AI Enhanced video milegi\n\n"
        "Bas file bhejo! ✅",
        parse_mode="Markdown"
    )

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    audio = update.message.audio or update.message.voice or update.message.document
    if not audio:
        return

    keyboard = [
        [InlineKeyboardButton("🇵🇰 Urdu", callback_data="LANG|ur"),
         InlineKeyboardButton("🇸🇦 Arabic", callback_data="LANG|ar")],
        [InlineKeyboardButton("🇬🇧 English", callback_data="LANG|en"),
         InlineKeyboardButton("🔍 Auto", callback_data="LANG|auto")]
    ]

    context.user_data["audio_file_id"] = audio.file_id

    await update.message.reply_text(
        "🌐 Kaunsi language hai?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    lang = query.data.split("|")[1]
    file_id = context.user_data.get("audio_file_id")

    if not file_id:
        await query.edit_message_text("❌ File nahi mili, dobara bhejo!")
        return

    lang_names = {"ur": "Urdu 🇵🇰", "ar": "Arabic 🇸🇦", "en": "English 🇬🇧", "auto": "Auto 🔍"}
    await query.edit_message_text(f"⏳ Transcribing ({lang_names.get(lang)})... sabr karo!")

    file = await context.bot.get_file(file_id)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        tmp_path = tmp.name

    srt_path = tmp_path + ".srt"
    txt_path = tmp_path + ".txt"

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, do_transcribe, tmp_path, lang
        )

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(result["srt"])
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["txt"])

        await query.edit_message_text(
            f"✅ Done!\n🌐 Language: {result['language']}\n📝 Segments: {result['segments']}"
        )

        with open(srt_path, "rb") as f:
            await query.message.reply_document(f, filename="subtitles.srt",
                                                caption="📄 SRT — Islamic Beacon")
        with open(txt_path, "rb") as f:
            await query.message.reply_document(f, filename="transcript.txt",
                                                caption="📄 Plain Text")
    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        await query.edit_message_text(f"❌ Error: {str(e)}")
    finally:
        for p in [tmp_path, srt_path, txt_path]:
            if os.path.exists(p):
                os.remove(p)

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    video = update.message.video or update.message.document
    if not video:
        return

    msg = await update.message.reply_text(
        "🎬 Video mil gayi!\n⏳ AI enhancement chal rahi hai...\n"
        "_(Free GPU slow hota hai — 10-20 min lag sakte hain)_",
        parse_mode="Markdown"
    )

    file = await video.get_file()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        await file.download_to_drive(tmp.name)
        input_path = tmp.name

    output_path = input_path.replace(".mp4", "_enhanced.mp4")

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, do_enhance, input_path, output_path)

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        await msg.edit_text(f"✅ Done! ({size_mb:.1f} MB) — Bhej raha hun...")

        with open(output_path, "rb") as f:
            await update.message.reply_video(
                f, caption="🎬 AI Enhanced — Islamic Beacon ✨"
            )
    except Exception as e:
        logger.error(f"Enhance error: {e}")
        await msg.edit_text(f"❌ Error: {str(e)}")
    finally:
        for p in [input_path, output_path]:
            if os.path.exists(p):
                os.remove(p)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN environment variable set nahi hai!")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(
        filters.AUDIO | filters.VOICE, handle_audio
    ))
    app.add_handler(MessageHandler(
        filters.VIDEO | filters.Document.VIDEO, handle_video
    ))
    app.add_handler(CallbackQueryHandler(
        handle_language_callback, pattern="^LANG\\|"
    ))

    print("🤖 Bot chal raha hai!")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
