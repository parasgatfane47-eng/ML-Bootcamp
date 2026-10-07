# -*- coding: utf-8 -*-
"""
=============================================================================
MEETING ASSISTANT — END-TO-END PIPELINE
=============================================================================

PROBLEM STATEMENT
-----------------
Take a raw meeting audio file and produce:
  1. A speaker-labelled, timestamped raw transcript  (WhisperX + diarization)
  2. A cleaned, corrected transcript                 (LLM1 — Gemini Flash Lite)
  3. Structured meeting output:
       • Minutes of Meeting
       • Decisions Made
       • Key Action Items               (LLM2 — Gemini Flash Lite + Qwen tokenizer)

PIPELINE STAGES
---------------
Stage 1 — Speech-to-Text (STT)
    Audio  ──► WhisperX transcription ──► word alignment ──► speaker diarization
    Output : raw timed transcript  [HH:MM - HH:MM] SPEAKER_XX: text

Stage 2 — Transcript Refinement (LLM1)
    Raw transcript ──► Gemini context-aware correction
    Output : cleaned, readable, timestamped transcript

Stage 3 — Meeting Analysis (LLM2)
    Refined transcript ──► Qwen-tokeniser chunking (60 k / 5 k overlap)
            ──► per-chunk analysis (Gemini)
            ──► context-safe reduction (if needed)
            ──► final structured JSON (Gemini)
    Output : { minutes_of_meeting, decisions_made, action_items }

REQUIREMENTS
------------
    pip install whisperx langchain-google-genai langchain google-genai
                pydantic transformers openai
    ffmpeg must be installed and on PATH.

SECRETS  (set as environment variables or Colab userdata)
    HF_TOKEN            — HuggingFace token for diarization models
    GEMINI_API_KEY      — Google AI Studio key
=============================================================================
"""

# ---------------------------------------------------------------------------
# 0.  STANDARD-LIBRARY IMPORTS
# ---------------------------------------------------------------------------
import gc
import json
import os
import re
import subprocess
import time
from dataclasses import replace

# ---------------------------------------------------------------------------
# 0a.  API KEY DEFAULTS
#      Priority: env variable  >  Colab userdata  >  hardcoded default below
#      ⚠️  Rotate these keys if they are ever exposed publicly.
# ---------------------------------------------------------------------------

_DEFAULT_KEYS: dict[str, str] = {
    "GEMINI_API_KEY": "AQ.Ab8RN6JUwBBey7pMpg81ExYLthJKKR1LO0O7th7KDFG4kWd4Ig",
    "HF_TOKEN":       "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",
}


def _get_secret(name: str) -> str | None:
    """
    Return a secret by priority:
      1. Environment variable
      2. Google Colab userdata
      3. Hardcoded default in _DEFAULT_KEYS
    """
    val = os.environ.get(name)
    if val:
        return val
    try:
        from google.colab import userdata
        colab_val = userdata.get(name)
        if colab_val:
            return colab_val
    except Exception:
        pass
    # Fall back to hardcoded defaults
    return _DEFAULT_KEYS.get(name)


# =============================================================================
# STAGE 1 — SPEECH-TO-TEXT  (WhisperX + diarization)
# =============================================================================

# ─── Constants ───────────────────────────────────────────────────────────────
ALLOWED_EXT = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg",
    ".opus", ".aac", ".wma", ".mp4", ".webm", ".mkv", ".mov"
}
MAX_MB = 500


class AudioError(Exception):
    """Raised with a user-friendly message when audio cannot be processed."""


# ─── Audio validation ─────────────────────────────────────────────────────────

def validate_audio(path: str, min_seconds: float = 1.0) -> float | None:
    """
    Check extension, size, and decodability.

    Returns duration in seconds, or None when the file is valid but its
    length cannot be determined from the header (e.g. streamed WAVs).
    Raises AudioError for unusable files.
    """
    if not path or not os.path.exists(path):
        raise AudioError("No file was provided.")

    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXT:
        raise AudioError(
            f"Unsupported file type '{ext or 'none'}'. "
            "Please upload one of: "
            + ", ".join(sorted(e.lstrip(".") for e in ALLOWED_EXT))
            + "."
        )

    size = os.path.getsize(path)
    if size == 0:
        raise AudioError("The file is empty (0 bytes).")
    if size > MAX_MB * 1024 * 1024:
        raise AudioError(
            f"The file is too large ({size / 1e6:.0f} MB). "
            f"Maximum is {MAX_MB} MB."
        )

    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "format=duration",
                "-of", "json", path,
            ],
            capture_output=True, text=True, timeout=60,
        )
    except FileNotFoundError:
        raise AudioError(
            "ffmpeg/ffprobe is not installed on this machine "
            "(in Colab run: !apt-get -y install ffmpeg)."
        )
    except subprocess.TimeoutExpired:
        raise AudioError("Timed out while reading the file. It may be corrupted.")

    if out.returncode != 0:
        why = (out.stderr or "").strip().splitlines()
        raise AudioError(
            "The file could not be read as audio. "
            "It may be corrupted or not a real audio file."
            + (f" (ffprobe: {why[-1][:200]})" if why else "")
        )

    try:
        duration = float(
            json.loads(out.stdout).get("format", {}).get("duration")
        )
    except (TypeError, ValueError):
        # Streamed WAV — try decoding 1 second
        test = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-t", "1", "-i", path,
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=60,
        )
        if test.returncode != 0 or test.stderr.strip():
            raise AudioError("No readable audio track was found in this file.")
        return None  # valid but length unknown

    if duration < min_seconds:
        raise AudioError(
            f"The audio is too short ({duration:.1f}s) to transcribe."
        )
    return duration


# ─── Turn-building helpers ────────────────────────────────────────────────────

_TERM = re.compile(r"[.?!…]+[\"')]*$")
_is_end = lambda w: bool(_TERM.search(w["w"]))


def _build_turns(result: dict, min_words: int = 2) -> list:
    """
    Group aligned words into speaker turns.

    Returns [[speaker, [word_dicts]], …].
    word_dict keys: w (text), s/e (word start/end), ss/se (segment fallback).
    """
    turns = []
    for seg in result["segments"]:
        last = seg.get("speaker", "UNKNOWN")
        ss, se = seg.get("start"), seg.get("end")
        words = seg.get("words") or [
            {"word": seg["text"].strip(), "speaker": last}
        ]
        for w in words:
            spk = w.get("speaker", last)
            last = spk
            item = {
                "w": w["word"],
                "s": w.get("start"),
                "e": w.get("end"),
                "ss": ss,
                "se": se,
            }
            if turns and turns[-1][0] == spk:
                turns[-1][1].append(item)
            else:
                turns.append([spk, [item]])

    # Smooth 1-word flickers:  A B A  →  A
    i = 1
    while i < len(turns) - 1:
        if (
            len(turns[i][1]) < min_words
            and turns[i - 1][0] == turns[i + 1][0]
        ):
            turns[i - 1][1] += turns[i][1] + turns[i + 1][1]
            del turns[i: i + 2]
        else:
            i += 1
    return turns


def _merge_same(turns: list) -> list:
    """Merge consecutive turns belonging to the same speaker."""
    out = []
    for s, w in turns:
        if not w:
            continue
        if out and out[-1][0] == s:
            out[-1][1] += w
        else:
            out.append([s, list(w)])
    return out


def _snap_boundaries(turns: list, max_move: int = 3) -> list:
    """Fix 1-3 word speaker flips that happen mid-sentence."""
    turns = _merge_same(turns)
    i = 0
    while i < len(turns) - 1:
        p, n = turns[i][1], turns[i + 1][1]
        moved = False
        if p and n and not _is_end(p[-1]):
            k = max([j for j, w in enumerate(p) if _is_end(w)], default=-1)
            F = p[k + 1:]
            c = next((j for j, w in enumerate(n) if _is_end(w)), None)
            C = n[: c + 1] if c is not None else None
            if C is not None and min(len(F), len(C)) <= max_move:
                if len(C) <= len(F):
                    turns[i][1] = p + C
                    turns[i + 1][1] = n[len(C):]
                else:
                    turns[i][1] = p[: k + 1]
                    turns[i + 1][1] = F + n
                moved = True
        if moved:
            turns = _merge_same(turns)
            i = max(0, i - 1)
        else:
            i += 1
    return turns


def _fmt_time(t: float) -> str:
    t = max(0, int(round(t)))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _to_records(turns: list) -> list[dict]:
    """Convert turns to list of {speaker, start, end, text} dicts."""
    recs = []
    for spk, words in turns:
        starts = [w["s"] for w in words if w["s"] is not None]
        ends   = [w["e"] for w in words if w["e"] is not None]
        start = (
            min(starts) if starts
            else next(
                (w["ss"] for w in words if w["ss"] is not None), 0.0
            )
        )
        end = (
            max(ends) if ends
            else next(
                (w["se"] for w in reversed(words) if w["se"] is not None),
                start,
            )
        )
        recs.append({
            "speaker": spk,
            "start": round(start, 2),
            "end": round(end, 2),
            "text": " ".join(w["w"] for w in words).strip(),
        })
    return recs


def _format_timed(recs: list[dict]) -> str:
    return "\n".join(
        f"[{_fmt_time(r['start'])} - {_fmt_time(r['end'])}] "
        f"{r['speaker']}: {r['text']}"
        for r in recs
    )


def _format_plain(recs: list[dict]) -> str:
    return "\n".join(f"[{r['speaker']}] {r['text']}" for r in recs)


# ─── Model cache (avoids reloading between calls in the same process) ─────────
_MODEL_CACHE: dict = {}


def _device():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return dev, ("float16" if dev == "cuda" else "int8")


def clear_stt_models():
    """Free GPU memory (call if you hit out-of-memory errors)."""
    _MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ─── Main STT entry point ─────────────────────────────────────────────────────

def transcribe_audio(
    path: str,
    context: str = "",
    num_speakers: int | None = None,
    min_speakers: int = 2,
    max_speakers: int = 8,
    model_size: str = "large-v3",
    hf_token: str | None = None,
    batch_size: int = 16,
    progress=print,
) -> dict:
    """
    Audio file → structured transcript dict.

    Returns
    -------
    {
        "records"  : [{speaker, start, end, text}, ...],
        "timed"    : "[MM:SS - MM:SS] SPEAKER: text\\n...",
        "plain"    : "[SPEAKER] text\\n...",
        "duration" : float | None,
        "diarized" : bool,
        "warnings" : [str, ...]
    }

    Raises AudioError with a user-friendly message for unusable audio.
    """
    import whisperx

    warnings = []

    def note(msg):
        warnings.append(msg)
        progress(f"⚠️  {msg}")

    progress("1/5  Checking the audio file...")
    duration = validate_audio(path)
    device, ctype = _device()

    progress(f"2/5  Transcribing ({model_size} on {device}) — this takes a while...")
    key = ("asr", model_size, device, ctype)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = whisperx.load_model(
            model_size, device, compute_type=ctype, language="en"
        )
    model = _MODEL_CACHE[key]
    model.options = replace(
        model.options,
        initial_prompt=(context.strip() or None),
    )
    try:
        audio = whisperx.load_audio(path)
    except Exception as e:
        raise AudioError(
            f"The audio could not be decoded: {str(e).strip()[:300]}"
        )
    result = model.transcribe(audio, batch_size=batch_size)
    if not result.get("segments"):
        raise AudioError("No speech was detected in this recording.")

    progress("3/5  Aligning words to timestamps...")
    try:
        akey = ("align", device)
        if akey not in _MODEL_CACHE:
            _MODEL_CACHE[akey] = whisperx.load_align_model(
                language_code="en", device=device
            )
        align_model, metadata = _MODEL_CACHE[akey]
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device
        )
    except Exception as e:
        note(
            f"Word alignment failed ({type(e).__name__}); "
            "timestamps will be segment-level."
        )

    progress("4/5  Identifying speakers...")
    diarized = False
    token = hf_token or _get_secret("HF_TOKEN")
    try:
        if not token:
            raise RuntimeError(
                "No HuggingFace token found "
                "(set the HF_TOKEN environment variable)."
            )
        from whisperx.diarize import DiarizationPipeline

        dkey = ("diar", device)
        if dkey not in _MODEL_CACHE:
            _MODEL_CACHE[dkey] = DiarizationPipeline(
                token=token, device=device
            )
        kw = (
            {"num_speakers": num_speakers}
            if num_speakers
            else {"min_speakers": min_speakers, "max_speakers": max_speakers}
        )
        segs = _MODEL_CACHE[dkey](audio, **kw)
        result = whisperx.assign_word_speakers(segs, result, fill_nearest=True)
        diarized = True
    except Exception as e:
        note(
            f"Speaker identification skipped: {e}. "
            "Output will not have speaker labels."
        )

    progress("5/5  Building the transcript...")
    if diarized:
        records = _to_records(_snap_boundaries(_build_turns(result)))
    else:
        records = [
            {
                "speaker": "UNKNOWN",
                "start": round(s.get("start", 0.0), 2),
                "end":   round(s.get("end",   0.0), 2),
                "text":  s["text"].strip(),
            }
            for s in result["segments"]
        ]

    return {
        "records":  records,
        "timed":    _format_timed(records),
        "plain":    _format_plain(records),
        "duration": duration,
        "diarized": diarized,
        "warnings": warnings,
    }


# ─── Transcript save helper ────────────────────────────────────────────────────

def save_stt_outputs(stt_result: dict, base: str = "transcript") -> list[str]:
    """
    Write <base>_timed.txt, <base>.txt (plain) and <base>.json.
    Returns the list of written file names.
    """
    files_map = {
        f"{base}_timed.txt": stt_result["timed"],
        f"{base}.txt":       stt_result["plain"],
        f"{base}.json":      json.dumps(stt_result["records"], indent=2),
    }
    for name, content in files_map.items():
        with open(name, "w", encoding="utf-8") as fh:
            fh.write(content)
    return list(files_map)


# =============================================================================
# STAGE 2 — TRANSCRIPT REFINEMENT  (LLM1 — Gemini via LangChain)
# =============================================================================

# ─── System prompt ────────────────────────────────────────────────────────────
_REFINEMENT_SYSTEM_PROMPT = """\
You are an expert transcript editor and context-aware transcription correction system.

You will receive a raw meeting transcript containing timestamps and speaker labels.

Your task is to produce a cleaned, corrected, highly readable version of the transcript \
while preserving the original conversation, chronology, speakers, timestamps, and meaning.

The transcript may contain speech-recognition errors. Use surrounding conversational \
context to identify and correct obvious transcription mistakes.

IMPORTANT RULES:

1. PRESERVE EVERY TIMESTAMP — keep every timestamp exactly as provided in [HH:MM:SS] format.
2. PRESERVE EVERY SPEAKER LABEL — keep speaker labels exactly as provided.
3. PRESERVE CHRONOLOGICAL ORDER.
4. PRESERVE THE ORIGINAL MEANING — do NOT summarize.
5. USE CONTEXT TO CORRECT OBVIOUS TRANSCRIPTION ERRORS (homophones, phonetic substitutions, etc.).
6. CONTEXT MAY EXTEND ACROSS MULTIPLE LINES.
7. CORRECT OBVIOUS SPEECH-TO-TEXT ERRORS (homophones, missing words, wrong names).
8. DO NOT GUESS WHEN MEANING IS UNCERTAIN — prefer original wording.
9. PRESERVE IMPORTANT INFORMATION (names, numbers, dates, decisions, action items).
10. REMOVE MEANINGLESS FILLERS (um, uh, er, you know) when they contribute nothing.
11. REMOVE ACCIDENTAL REPETITIONS (stutters, false starts).
12. PRESERVE MEANINGFUL SHORT RESPONSES (Okay. Right. Yes. No. Exactly.).
13. CORRECT GRAMMAR WHEN INTENDED MEANING IS CLEAR — keep natural spoken style.
14. PRESERVE NAMES AND IDENTITIES CAREFULLY.
15. NEVER ADD NEW INFORMATION not present in the original transcript.
16. MAINTAIN ONE OUTPUT LINE PER INPUT TRANSCRIPT ENTRY.

OUTPUT FORMAT — return ONLY the corrected transcript, nothing else:
[HH:MM:SS] Speaker X: Corrected speech
[HH:MM:SS] Speaker Y: Corrected speech
"""

_REFINEMENT_HUMAN_PROMPT = """\
Clean and correct the following raw meeting transcript using the rules above.

Use the surrounding dialogue as context when identifying speech-to-text errors.

Raw transcript:

{transcript}
"""


def refine_transcript(
    raw_transcript: str,
    gemini_api_key: str | None = None,
    model_name: str = "gemini-2.0-flash-lite",
) -> str:
    """
    Stage 2: raw timestamped transcript → cleaned, corrected transcript.

    Uses Gemini via LangChain (langchain-google-genai).

    Parameters
    ----------
    raw_transcript : str
        The timed transcript produced by Stage 1.
    gemini_api_key : str, optional
        Gemini API key.  Falls back to the GEMINI_API_KEY env variable.
    model_name : str
        Gemini model to use.

    Returns
    -------
    str
        The refined transcript text.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate

    api_key = gemini_api_key or _get_secret("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Pass it as gemini_api_key= or set the environment variable."
        )
    os.environ["GOOGLE_API_KEY"] = api_key  # LangChain picks this up

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _REFINEMENT_SYSTEM_PROMPT),
        ("human",  _REFINEMENT_HUMAN_PROMPT),
    ])

    chain = prompt | llm
    response = chain.invoke({"transcript": raw_transcript})

    # Handle both plain string and structured content responses
    content = response.content
    if isinstance(content, list):
        # Gemini sometimes returns a list of content blocks
        content = content[0].get("text", "") if content else ""

    return content.strip()


# =============================================================================
# STAGE 3 — MEETING ANALYSIS  (LLM2 — Gemini + Qwen tokeniser)
# =============================================================================

# ─── Pydantic output schemas ──────────────────────────────────────────────────
try:
    from pydantic import BaseModel, Field, ValidationError

    class ActionItem(BaseModel):
        """One actionable task identified from the meeting."""
        action: str = Field(
            description="The specific action or task that needs to be done."
        )
        assigned_to: str | None = Field(
            default=None,
            description="Person responsible. null if not specified.",
        )
        deadline: str | None = Field(
            default=None,
            description="Deadline / due date. null if not specified.",
        )

    class MeetingOutput(BaseModel):
        """Final structured output of the meeting analysis pipeline."""
        minutes_of_meeting: str = Field(
            description=(
                "Clear and concise minutes of the meeting covering "
                "important discussions, decisions, and conclusions."
            )
        )
        decisions_made: list[str] = Field(
            default_factory=list,
            description="Decisions explicitly made during the meeting.",
        )
        action_items: list[ActionItem] = Field(
            default_factory=list,
            description="Concrete action items identified from the meeting.",
        )

except ImportError:
    raise ImportError("Install pydantic:  pip install pydantic")


# ─── Tokenizer (Qwen) ─────────────────────────────────────────────────────────
# Loaded lazily so that the module can be imported without downloading weights.
_TOKENIZER = None
_TOKENIZER_NAME = "Qwen/Qwen2.5-72B-Instruct"

# Chunk settings (defined by the problem statement)
CHUNK_SIZE    = 60_000   # tokens per chunk
CHUNK_OVERLAP =  5_000   # overlap between consecutive chunks

FINAL_INPUT_TOKEN_LIMIT     = 100_000
REDUCTION_BATCH_TOKEN_LIMIT =  50_000


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(
            _TOKENIZER_NAME, trust_remote_code=True
        )
    return _TOKENIZER


def _count_tokens(text: str) -> int:
    tok = _get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


# ─── Gemini client (google-genai) ─────────────────────────────────────────────
_GENAI_CLIENT = None
_LLM2_MODEL   = "gemini-2.0-flash-lite"


def _get_genai_client(api_key: str | None = None):
    global _GENAI_CLIENT
    if _GENAI_CLIENT is None:
        from google import genai
        key = api_key or _get_secret("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Pass it as gemini_api_key= or set the environment variable."
            )
        _GENAI_CLIENT = genai.Client(api_key=key)
    return _GENAI_CLIENT


def _llm2_generate(prompt: str, api_key: str | None = None) -> str:
    """Send a prompt to Gemini and return the response text."""
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    client = _get_genai_client(api_key)
    response = client.models.generate_content(
        model=_LLM2_MODEL,
        contents=prompt,
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    return response.text


# ─── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_transcript(
    transcript: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> dict:
    """
    Tokenize the transcript (Qwen tokeniser) and split into overlapping chunks.

    Returns
    -------
    {
        "total_tokens": int,
        "total_chunks": int,
        "chunk_size":   int,
        "overlap":      int,
        "chunks":       [{"chunk_id", "start_token", "end_token",
                          "token_count", "text"}, ...]
    }
    """
    tok = _get_tokenizer()
    token_ids = tok.encode(transcript, add_special_tokens=False)

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")
    step = chunk_size - overlap

    raw_chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        raw_chunks.append(token_ids[start:end])
        if end >= len(token_ids):
            break
        start += step

    chunks = []
    for i, raw in enumerate(raw_chunks):
        text = tok.decode(raw, skip_special_tokens=True)
        start_tok = i * step
        chunks.append({
            "chunk_id":    i + 1,
            "start_token": start_tok,
            "end_token":   start_tok + len(raw),
            "token_count": len(raw),
            "text":        text,
        })

    return {
        "total_tokens": len(token_ids),
        "total_chunks": len(chunks),
        "chunk_size":   chunk_size,
        "overlap":      overlap,
        "chunks":       chunks,
    }


# ─── Per-chunk analysis prompt ─────────────────────────────────────────────────

_CHUNK_ANALYSIS_PROMPT = """\
You are analyzing one section of a meeting transcript.

This section belongs to a larger meeting.

Extract information that will later be used to create:

1. Minutes of Meeting
2. Decisions Made
3. Key Action Items

Focus on:

1. Important discussions
2. Explicitly made decisions
3. Explicit tasks or commitments
4. Explicitly assigned responsibilities
5. Explicit deadlines

IMPORTANT ACTION ITEM RULE:
Only report something as a potential action item if the transcript explicitly
states that someone will do something, has agreed to do something, or has been
assigned a task.
Do NOT treat suggestions, ideas, opinions, questions, or general discussion
as action items.

IMPORTANT DECISION RULE:
Only report something as a decision if the transcript indicates it was actually
decided, agreed, approved, rejected, finalized, or confirmed.
Do not convert suggestions or discussions into decisions.

Only use information explicitly present in the transcript.
Do NOT invent names, responsibilities, deadlines, decisions, tasks, or facts.
Do NOT produce the final JSON.

Transcript section:

{chunk}
"""


def _process_all_chunks(chunked: dict, api_key: str | None = None) -> dict:
    """Send each text chunk to Gemini and collect the analyses."""
    processed = []
    total = len(chunked["chunks"])

    for chunk in chunked["chunks"]:
        cid = chunk["chunk_id"]
        print(f"  Analyzing chunk {cid}/{total}...")

        prompt = _CHUNK_ANALYSIS_PROMPT.format(chunk=chunk["text"])
        try:
            output = _llm2_generate(prompt, api_key)
            if not output or not output.strip():
                raise ValueError("Empty response from Gemini.")
            processed.append({
                **{k: chunk[k] for k in
                   ("chunk_id", "start_token", "end_token", "token_count")},
                "llm_output": output,
                "status":     "success",
            })
        except Exception as e:
            print(f"  ❌  Chunk {cid} failed: {e}")
            raise RuntimeError(
                f"Gemini failed while processing chunk {cid}: {e}"
            ) from e

    return {
        "total_tokens": chunked["total_tokens"],
        "total_chunks": total,
        "chunks":       processed,
    }


def _combine_chunk_outputs(processed: dict) -> str:
    outputs = []
    for chunk in processed["chunks"]:
        out = chunk.get("llm_output")
        if out and str(out).strip():
            outputs.append(
                f"===== CHUNK ANALYSIS {chunk['chunk_id']} =====\n{out}"
            )
    if not outputs:
        raise ValueError("No successful chunk outputs available.")
    return "\n\n".join(outputs)


# ─── Context-safe reduction ────────────────────────────────────────────────────

_REDUCTION_PROMPT = """\
You are condensing analyses from different sections of the SAME meeting.

Preserve all information needed for:
1. Minutes of Meeting
2. Action Items

Preserve:
- Important discussions
- Decisions
- Conclusions
- Tasks, responsibilities, and deadlines
- Important context

Remove only:
- Repetition
- Duplicate information

Do NOT invent information.
Return only the condensed meeting analysis.

Analyses:

{analyses}
"""


def _make_context_safe(combined: str, api_key: str | None = None) -> str:
    """
    Iteratively reduce combined_analyses until it fits within
    FINAL_INPUT_TOKEN_LIMIT tokens.
    """
    tok = _get_tokenizer()
    current = combined

    while _count_tokens(current) > FINAL_INPUT_TOKEN_LIMIT:
        token_ids = tok.encode(current, add_special_tokens=False)
        batches, start = [], 0
        while start < len(token_ids):
            end = min(start + REDUCTION_BATCH_TOKEN_LIMIT, len(token_ids))
            batches.append(
                tok.decode(token_ids[start:end], skip_special_tokens=True)
            )
            start = end

        reduced = []
        for i, batch in enumerate(batches, 1):
            print(f"  Reducing batch {i}/{len(batches)}...")
            r = _llm2_generate(
                _REDUCTION_PROMPT.format(analyses=batch), api_key
            )
            if r:
                reduced.append(r)

        current = "\n\n".join(
            f"===== REDUCED ANALYSIS {i} =====\n{out}"
            for i, out in enumerate(reduced, 1)
        )

    return current


# ─── Final output prompt ───────────────────────────────────────────────────────

_FINAL_OUTPUT_PROMPT = """\
You are the final meeting-analysis stage.

You will receive analyses from multiple sections of the SAME meeting transcript.

Produce ONLY the following three sections as valid JSON:

1. Minutes of Meeting
2. Decisions Made
3. Key Action Items

============================================================
ACTION ITEM RULES — VERY IMPORTANT
============================================================

An action item is ONLY a task that someone explicitly:
  - agreed to do
  - was explicitly assigned to
  - explicitly committed to doing

DO NOT create an action item from general discussion, suggestions, opinions,
questions, or information sharing.

Each distinct task should appear ONCE.

============================================================
DECISION RULES
============================================================

A decision is something explicitly: decided / agreed / approved / rejected /
finalized / confirmed as the chosen approach.

DO NOT treat a suggestion, discussion, possibility, or opinion as a decision.

============================================================
GENERAL RULES
============================================================

- Combine information across all sections; remove duplicates.
- Do not invent information.
- Do not assume responsibility unless explicitly stated → use null.
- Do not invent deadlines → use null.
- If no explicit decisions → return [].
- If no explicit action items → return [].

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON — no Markdown, no ```json, no explanations outside JSON.

{
  "minutes_of_meeting": "string",
  "decisions_made": ["decision 1", "decision 2"],
  "action_items": [
    {
      "action": "specific task",
      "assigned_to": "person or null",
      "deadline": "deadline or null"
    }
  ]
}

============================================================
MEETING ANALYSES
============================================================

{combined_analyses}
"""


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def _generate_final_output(safe_analyses: str, api_key: str | None = None) -> str:
    prompt = _FINAL_OUTPUT_PROMPT.replace("{combined_analyses}", safe_analyses)
    return _llm2_generate(prompt, api_key)


def _validate_final_output(raw: str) -> MeetingOutput:
    raw = _strip_markdown_fences(raw)
    try:
        return MeetingOutput.model_validate_json(raw)
    except ValidationError as e:
        raise ValueError(f"Final output failed Pydantic validation:\n{e}")


# ─── LLM2 main entry point ────────────────────────────────────────────────────

def analyze_transcript(
    refined_transcript: str,
    gemini_api_key: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> MeetingOutput:
    """
    Stage 3: refined transcript → structured meeting output.

    Parameters
    ----------
    refined_transcript : str
        The cleaned transcript produced by Stage 2.
    gemini_api_key : str, optional
        Gemini API key.  Falls back to GEMINI_API_KEY env variable.
    chunk_size : int
        Qwen token chunk size (default 60 000).
    chunk_overlap : int
        Token overlap between consecutive chunks (default 5 000).

    Returns
    -------
    MeetingOutput
        Pydantic object with .minutes_of_meeting, .decisions_made,
        and .action_items.
    """
    if not isinstance(refined_transcript, str):
        raise TypeError("refined_transcript must be a string.")
    if not refined_transcript.strip():
        raise ValueError("refined_transcript cannot be empty.")

    print("  Step 1/5  Chunking transcript...")
    chunked = _chunk_transcript(refined_transcript, chunk_size, chunk_overlap)
    print(
        f"            Total tokens : {chunked['total_tokens']:,}\n"
        f"            Total chunks : {chunked['total_chunks']}"
    )

    print("  Step 2/5  Analyzing chunks...")
    processed = _process_all_chunks(chunked, gemini_api_key)

    print("  Step 3/5  Combining chunk analyses...")
    combined = _combine_chunk_outputs(processed)

    print("  Step 4/5  Context-safety check...")
    safe = _make_context_safe(combined, gemini_api_key)
    print(f"            Final analysis tokens: {_count_tokens(safe):,}")

    print("  Step 5/5  Generating final structured output...")
    raw_output = _generate_final_output(safe, gemini_api_key)

    return _validate_final_output(raw_output)


# =============================================================================
# COMPLETE END-TO-END PIPELINE
# =============================================================================

def run_meeting_assistant(
    audio_path: str,
    context: str = "",
    hf_token: str | None = None,
    gemini_api_key: str | None = None,
    output_base: str = "meeting_output",
    num_speakers: int | None = None,
    min_speakers: int = 2,
    max_speakers: int = 8,
    model_size: str = "large-v3",
    batch_size: int = 16,
) -> dict:
    """
    Full pipeline:  audio file  →  structured meeting output.

    Parameters
    ----------
    audio_path : str
        Path to the audio/video file.
    context : str
        Optional hint for the ASR model
        (e.g. "Participants: Alice, Bob. Topic: product roadmap.").
    hf_token : str, optional
        HuggingFace token for speaker diarization.
        Falls back to the HF_TOKEN environment variable.
    gemini_api_key : str, optional
        Google AI Studio key.
        Falls back to the GEMINI_API_KEY environment variable.
    output_base : str
        Base name for output files (no extension).
    num_speakers : int, optional
        Exact number of speakers (use instead of min/max if known).
    min_speakers : int
        Minimum number of speakers for diarization.
    max_speakers : int
        Maximum number of speakers for diarization.
    model_size : str
        WhisperX model size (default "large-v3").
    batch_size : int
        Transcription batch size.

    Returns
    -------
    {
        "stt"          : STT result dict,
        "refined"      : refined transcript string,
        "analysis"     : MeetingOutput pydantic object,
        "output_files" : list of written file paths
    }
    """

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 1 — SPEECH-TO-TEXT")
    print("=" * 70)

    stt_result = transcribe_audio(
        path=audio_path,
        context=context,
        hf_token=hf_token,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        model_size=model_size,
        batch_size=batch_size,
    )

    if stt_result["warnings"]:
        for w in stt_result["warnings"]:
            print(f"⚠️  {w}")

    print("\nRaw transcript preview (first 500 chars):\n")
    print(stt_result["timed"][:500])

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 2 — TRANSCRIPT REFINEMENT")
    print("=" * 70)

    refined = refine_transcript(
        raw_transcript=stt_result["timed"],
        gemini_api_key=gemini_api_key,
    )

    print("\nRefined transcript preview (first 500 chars):\n")
    print(refined[:500])

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 3 — MEETING ANALYSIS")
    print("=" * 70)

    analysis = analyze_transcript(
        refined_transcript=refined,
        gemini_api_key=gemini_api_key,
    )

    # ── Display results ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("1. MINUTES OF MEETING")
    print("=" * 70)
    print(analysis.minutes_of_meeting)

    print("\n" + "=" * 70)
    print("2. DECISIONS MADE")
    print("=" * 70)
    if analysis.decisions_made:
        for i, d in enumerate(analysis.decisions_made, 1):
            print(f"{i}. {d}")
    else:
        print("No explicit decisions identified.")

    print("\n" + "=" * 70)
    print("3. KEY ACTION ITEMS")
    print("=" * 70)
    if analysis.action_items:
        for i, item in enumerate(analysis.action_items, 1):
            print(f"\n{i}. {item.action}")
            print(f"   Assigned to : {item.assigned_to or 'Not specified'}")
            print(f"   Deadline    : {item.deadline    or 'Not specified'}")
    else:
        print("No action items identified.")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING OUTPUT FILES")
    print("=" * 70)

    written_files = []

    # 1. Raw timed transcript
    f1 = f"{output_base}_raw_timed.txt"
    with open(f1, "w", encoding="utf-8") as fh:
        fh.write(stt_result["timed"])
    written_files.append(f1)

    # 2. Raw plain transcript
    f2 = f"{output_base}_raw.txt"
    with open(f2, "w", encoding="utf-8") as fh:
        fh.write(stt_result["plain"])
    written_files.append(f2)

    # 3. Raw transcript JSON (records with timestamps)
    f3 = f"{output_base}_raw.json"
    with open(f3, "w", encoding="utf-8") as fh:
        json.dump(stt_result["records"], fh, indent=2)
    written_files.append(f3)

    # 4. Refined transcript
    f4 = f"{output_base}_refined.txt"
    with open(f4, "w", encoding="utf-8") as fh:
        fh.write(refined)
    written_files.append(f4)

    # 5. Structured meeting output JSON
    result_dict = analysis.model_dump()
    f5 = f"{output_base}_analysis.json"
    with open(f5, "w", encoding="utf-8") as fh:
        json.dump(result_dict, fh, indent=2, ensure_ascii=False)
    written_files.append(f5)

    # 6. Human-readable meeting output TXT
    f6 = f"{output_base}_analysis.txt"
    with open(f6, "w", encoding="utf-8") as fh:
        fh.write("1. MINUTES OF MEETING\n")
        fh.write("=" * 70 + "\n\n")
        fh.write(analysis.minutes_of_meeting)
        fh.write("\n\n\n")

        fh.write("2. DECISIONS MADE\n")
        fh.write("=" * 70 + "\n\n")
        if analysis.decisions_made:
            for i, d in enumerate(analysis.decisions_made, 1):
                fh.write(f"{i}. {d}\n")
        else:
            fh.write("No explicit decisions identified.\n")
        fh.write("\n\n")

        fh.write("3. KEY ACTION ITEMS\n")
        fh.write("=" * 70 + "\n\n")
        if analysis.action_items:
            for i, item in enumerate(analysis.action_items, 1):
                fh.write(f"{i}. {item.action}\n")
                fh.write(
                    f"   Assigned to : "
                    f"{item.assigned_to or 'Not specified'}\n"
                )
                fh.write(
                    f"   Deadline    : "
                    f"{item.deadline    or 'Not specified'}\n\n"
                )
        else:
            fh.write("No action items identified.\n")
    written_files.append(f6)

    for path in written_files:
        print(f"  ✅  {path}")

    return {
        "stt":          stt_result,
        "refined":      refined,
        "analysis":     analysis,
        "output_files": written_files,
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Meeting Assistant — end-to-end pipeline: "
            "audio  ─►  transcript  ─►  refinement  ─►  minutes + actions"
        )
    )
    parser.add_argument(
        "audio",
        help="Path to the audio / video file.",
    )
    parser.add_argument(
        "--context",
        default="",
        help=(
            "Optional context hint for the ASR model "
            "(e.g. 'Participants: Alice, Bob. Topic: roadmap.')."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (falls back to HF_TOKEN env var).",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Google Gemini API key (falls back to GEMINI_API_KEY env var).",
    )
    parser.add_argument(
        "--output-base",
        default="meeting_output",
        help="Base name for output files (default: meeting_output).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact number of speakers (optional).",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=2,
        help="Minimum speakers for diarization (default: 2).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=8,
        help="Maximum speakers for diarization (default: 8).",
    )
    parser.add_argument(
        "--model-size",
        default="large-v3",
        help="WhisperX model size (default: large-v3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="WhisperX transcription batch size (default: 16).",
    )

    args = parser.parse_args()

    run_meeting_assistant(
        audio_path=args.audio,
        context=args.context,
        hf_token=args.hf_token,
        gemini_api_key=args.gemini_api_key,
        output_base=args.output_base,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        model_size=args.model_size,
        batch_size=args.batch_size,
    )
