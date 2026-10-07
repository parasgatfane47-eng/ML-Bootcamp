"""
Stage 1 of the meeting assistant: audio file -> speaker-labelled, timestamped raw transcript.

    result = transcribe_audio("meeting.wav", context="Participants: ...")
    result["timed"]    # "[00:55 - 01:17] SPEAKER_03: text ..."
    result["plain"]    # "[SPEAKER_03] text ..."          (used by eval_transcript.py)
    result["records"]  # [{"speaker","start","end","text"}, ...]   (machine-readable)
"""
import gc, json, os, re, subprocess
from dataclasses import replace

# ----------------------------------------------------------------- errors / validation
ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".webm", ".mkv", ".mov"}
MAX_MB = 500

class AudioError(Exception):
    """Raised with a user-friendly message when the audio cannot be processed."""

def validate_audio(path, min_seconds=1.0):
    """Check extension, size and decodability. Returns duration in seconds or raises AudioError."""
    if not path or not os.path.exists(path):
        raise AudioError("No file was provided.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXT:
        raise AudioError(f"Unsupported file type '{ext or 'none'}'. Please upload one of: "
                         + ", ".join(sorted(e.lstrip('.') for e in ALLOWED_EXT)) + ".")
    size = os.path.getsize(path)
    if size == 0:
        raise AudioError("The file is empty (0 bytes).")
    if size > MAX_MB * 1024 * 1024:
        raise AudioError(f"The file is too large ({size/1e6:.0f} MB). Maximum is {MAX_MB} MB.")
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "format=duration",
             "-of", "json", path], capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        raise AudioError("ffmpeg/ffprobe is not installed on this machine "
                         "(in Colab run: !apt-get -y install ffmpeg).")
    except subprocess.TimeoutExpired:
        raise AudioError("Timed out while reading the file. It may be corrupted.")
    if out.returncode != 0:
        why = (out.stderr or "").strip().splitlines()
        raise AudioError("The file could not be read as audio. It may be corrupted or not a real audio file."
                         + (f" (ffprobe: {why[-1][:200]})" if why else ""))
    try:
        duration = float(json.loads(out.stdout).get("format", {}).get("duration"))
    except (TypeError, ValueError):
        # Some valid files (e.g. streamed WAVs) have no duration in the header: try decoding 1 second instead.
        test = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-t", "1", "-i", path, "-f", "null", "-"],
                              capture_output=True, text=True, timeout=60)
        if test.returncode != 0 or test.stderr.strip():
            raise AudioError("No readable audio track was found in this file.")
        return None                                   # valid, but length unknown
    if duration < min_seconds:
        raise AudioError(f"The audio is too short ({duration:.1f}s) to transcribe.")
    return duration

# ----------------------------------------------------------------- turn building
TERM = re.compile(r"[.?!…]+[\"')]*$")
is_end = lambda w: bool(TERM.search(w["w"]))

def build_turns(result, min_words=2):
    """Group aligned words into speaker turns: [[speaker, [word dicts]], ...]
    word dict: w=text, s/e=word start/end (None if unaligned), ss/se=segment start/end (fallback)."""
    turns = []
    for seg in result["segments"]:
        last = seg.get("speaker", "UNKNOWN")
        ss, se = seg.get("start"), seg.get("end")
        words = seg.get("words") or [{"word": seg["text"].strip(), "speaker": last}]
        for w in words:
            spk = w.get("speaker", last)             # unaligned tokens inherit previous speaker
            last = spk
            item = {"w": w["word"], "s": w.get("start"), "e": w.get("end"), "ss": ss, "se": se}
            if turns and turns[-1][0] == spk: turns[-1][1].append(item)
            else: turns.append([spk, [item]])
    i = 1                                             # smooth 1-word flickers: A B A -> A
    while i < len(turns) - 1:
        if len(turns[i][1]) < min_words and turns[i-1][0] == turns[i+1][0]:
            turns[i-1][1] += turns[i][1] + turns[i+1][1]
            del turns[i:i+2]
        else:
            i += 1
    return turns

def merge_same(turns):
    out = []
    for s, w in turns:
        if not w: continue
        if out and out[-1][0] == s: out[-1][1] += w
        else: out.append([s, list(w)])
    return out

def snap_boundaries(turns, max_move=3):
    """Fix 1-3 word speaker flips that happen mid-sentence."""
    turns = merge_same(turns); i = 0
    while i < len(turns) - 1:
        p, n = turns[i][1], turns[i+1][1]; moved = False
        if p and n and not is_end(p[-1]):
            k = max([j for j, w in enumerate(p) if is_end(w)], default=-1)
            F = p[k+1:]
            c = next((j for j, w in enumerate(n) if is_end(w)), None)
            C = n[:c+1] if c is not None else None
            if C is not None and min(len(F), len(C)) <= max_move:
                if len(C) <= len(F): turns[i][1] = p + C; turns[i+1][1] = n[len(C):]
                else:                turns[i][1] = p[:k+1]; turns[i+1][1] = F + n
                moved = True
        if moved: turns = merge_same(turns); i = max(0, i-1)
        else: i += 1
    return turns

def fmt_time(t):
    t = max(0, int(round(t)))
    h, rem = divmod(t, 3600); m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def to_records(turns):
    recs = []
    for spk, words in turns:
        starts = [w["s"] for w in words if w["s"] is not None]
        ends   = [w["e"] for w in words if w["e"] is not None]
        start = min(starts) if starts else next((w["ss"] for w in words if w["ss"] is not None), 0.0)
        end   = max(ends)   if ends   else next((w["se"] for w in reversed(words) if w["se"] is not None), start)
        recs.append({"speaker": spk, "start": round(start, 2), "end": round(end, 2),
                     "text": " ".join(w["w"] for w in words).strip()})
    return recs

def format_timed(recs):
    return "\n".join(f"[{fmt_time(r['start'])} - {fmt_time(r['end'])}] {r['speaker']}: {r['text']}" for r in recs)

def format_plain(recs):
    return "\n".join(f"[{r['speaker']}] {r['text']}" for r in recs)

# ----------------------------------------------------------------- models (cached between runs)
_CACHE = {}

def get_hf_token():
    tok = os.environ.get("HF_TOKEN")
    if tok: return tok
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return None

def _device():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return dev, ("float16" if dev == "cuda" else "int8")

def clear_models():
    """Free GPU memory (e.g. if you hit out-of-memory errors)."""
    _CACHE.clear(); gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception:
        pass

# ----------------------------------------------------------------- main entry point
def transcribe_audio(path, context="", num_speakers=None, min_speakers=2, max_speakers=8,
                     model_size="large-v3", hf_token=None, batch_size=16, progress=print):
    """Audio file -> {'records','timed','plain','duration','diarized','warnings'}.
    Raises AudioError (with a user-friendly message) for unusable audio."""
    import whisperx
    warnings = []
    def note(msg): warnings.append(msg); progress(f"⚠️  {msg}")

    progress("1/5  Checking the audio file...")
    duration = validate_audio(path)
    device, ctype = _device()

    progress(f"2/5  Transcribing ({model_size} on {device}) — this takes a while...")
    key = ("asr", model_size, device, ctype)
    if key not in _CACHE:
        _CACHE[key] = whisperx.load_model(model_size, device, compute_type=ctype, language="en")
    model = _CACHE[key]
    model.options = replace(model.options, initial_prompt=(context.strip() or None))  # per-recording context
    try:
        audio = whisperx.load_audio(path)
    except Exception as e:
        raise AudioError(f"The audio could not be decoded: {str(e).strip()[:300]}")
    result = model.transcribe(audio, batch_size=batch_size)
    if not result.get("segments"):
        raise AudioError("No speech was detected in this recording.")

    progress("3/5  Aligning words to timestamps...")
    try:
        akey = ("align", device)
        if akey not in _CACHE:
            _CACHE[akey] = whisperx.load_align_model(language_code="en", device=device)
        align_model, metadata = _CACHE[akey]
        result = whisperx.align(result["segments"], align_model, metadata, audio, device)
    except Exception as e:
        note(f"Word alignment failed ({type(e).__name__}); timestamps will be segment-level.")

    progress("4/5  Identifying speakers...")
    diarized = False
    token = hf_token or get_hf_token()
    try:
        if not token:
            raise RuntimeError("no Hugging Face token found (set the HF_TOKEN secret)")
        from whisperx.diarize import DiarizationPipeline
        dkey = ("diar", device)
        if dkey not in _CACHE:
            _CACHE[dkey] = DiarizationPipeline(token=token, device=device)
        kw = {"num_speakers": num_speakers} if num_speakers else {"min_speakers": min_speakers, "max_speakers": max_speakers}
        segs = _CACHE[dkey](audio, **kw)
        result = whisperx.assign_word_speakers(segs, result, fill_nearest=True)
        diarized = True
    except Exception as e:
        note(f"Speaker identification skipped: {e}. Output will not have speaker labels.")

    progress("5/5  Building the transcript...")
    if diarized:
        records = to_records(snap_boundaries(build_turns(result)))
    else:   # no speakers: one record per segment
        records = [{"speaker": "UNKNOWN", "start": round(s.get("start", 0.0), 2), "end": round(s.get("end", 0.0), 2),
                    "text": s["text"].strip()} for s in result["segments"]]
    return {"records": records, "timed": format_timed(records), "plain": format_plain(records),
            "duration": duration, "diarized": diarized, "warnings": warnings}

def save_outputs(res, base="transcript"):
    """Writes <base>_timed.txt, <base>.txt (plain) and <base>.json. Returns the file names."""
    files_ = {f"{base}_timed.txt": res["timed"], f"{base}.txt": res["plain"],
              f"{base}.json": json.dumps(res["records"], indent=2)}
    for name, content in files_.items():
        with open(name, "w", encoding="utf-8") as f: f.write(content)
    return list(files_)

# ----------------------------------------------------------------- Colab helpers
def upload_audio():
    """Colab: opens the upload dialog and returns the path of the uploaded file."""
    from google.colab import files
    uploaded = files.upload()
    if not uploaded:
        raise AudioError("No file was uploaded (the dialog was closed or the upload failed). Run the cell again.")
    name = next(iter(uploaded))
    if len(uploaded) > 1:
        print(f"Several files uploaded; using the first one: {name}")
    return name

def diagnose_audio(path):
    """Prints everything relevant about a file that is being rejected."""
    print("path     :", repr(path), "| exists:", os.path.exists(path or ""))
    if not path or not os.path.exists(path): return
    print("extension:", repr(os.path.splitext(path)[1].lower()), "| allowed:", os.path.splitext(path)[1].lower() in ALLOWED_EXT)
    print("size     :", os.path.getsize(path), "bytes")
    with open(path, "rb") as f: print("first 12 bytes:", f.read(12))
    print("   (a real WAV starts with b'RIFF....WAVE')")
    for tool in ("ffprobe", "ffmpeg"):
        try:
            v = subprocess.run([tool, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
            print(f"{tool:8s}:", v)
        except FileNotFoundError:
            print(f"{tool:8s}: NOT INSTALLED")
    r = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", path], capture_output=True, text=True)
    print("ffprobe return code:", r.returncode); print("ffprobe stderr:", r.stderr.strip() or "(none)")
    print("ffprobe stdout (first lines):\n" + "\n".join(r.stdout.splitlines()[:12]))
    try:
        print("validate_audio ->", validate_audio(path))
    except AudioError as e:
        print("validate_audio -> AudioError:", e)
