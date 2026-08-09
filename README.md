# ComfyUI — OpenAI Compatible LLM

One node that talks to **any** OpenAI-compatible endpoint (Mammouth, OpenRouter, Groq, Together,
LM Studio, Ollama, vLLM, llama.cpp, ...). It fetches the model list from the endpoint itself and
takes as many text inputs as you plug into it.

Node: **OpenAI Compatible LLM** (category `api/text`).

## Usage

1. Add the node and set `base_url`, e.g. `https://api.mammouth.ai/v1`
   (the part *before* `/models` and `/chat/completions`).
2. Paste your key into `api_key` (`sk-...`).
3. Press **Refresh models**. The `model` dropdown is filled from `GET {base_url}/models`.
   It also refreshes automatically when you change the URL or key, and when a workflow is loaded.
4. Connect one or more STRING outputs to `text_1`, `text_2`, ... — a new slot appears each time
   you connect one, up to 16.
5. The `text` output is the model's answer.

## Inputs

| Input | Notes |
|---|---|
| `base_url` | Endpoint root. Trailing `/` and a missing `https://` are handled. |
| `api_key` | See *API keys* below. |
| `model` | Populated by **Refresh models**. |
| `text_1...text_16` | Prompt parts. Empty ones are skipped. Slots grow as you connect them. |
| `system_prompt` | Optional system message. |
| `input_mode` | `join` (default): all texts become one user message. `separate_messages`: one user message per input. |
| `separator` | Used by `join` mode. `\n` and `\t` become real newlines/tabs. Default `\n\n`. |
| `temperature` | `-1` omits the field from the request (useful for reasoning models that reject it). |
| `max_tokens` | `0` leaves it to the provider. |
| `timeout` | Seconds to wait for the response. |
| `seed` | Never sent to the API — it only decides whether the node re-runs or returns its cached answer. Set *control after generate* to `randomize` to always call the API. |
| `reuse_last_result` | **On**: output the answer from last time, without calling the API. **Off**: generate normally. See below. |
| `images` | Optional IMAGE input. Every frame of the batch is sent as its own image. Needs a vision model. |
| `video` | Optional VIDEO input, inlined as a `video_url` block. Provider-specific — see below. |
| `image_detail` | OpenAI `detail` hint: `auto`, `low` (much cheaper), `high` (reads fine print). |
| `image_format` | `jpeg` (default), `png` (lossless, better for screenshots and text), `webp`. |
| `image_max_side` | Downscale so the longest side is at most this many pixels. `0` sends full size. |
| `video_max_mb` | Refuse to upload a video bigger than this, instead of failing at the provider. Default 20. |

## Images and video

Connect an IMAGE to `images` and the prompt is sent as OpenAI-style content blocks — a text block
followed by one `image_url` block per frame, as a `data:` URL. A batch of 4 becomes 4 images in one
message. With `input_mode = separate_messages` the media is attached to the **last** user message.

With nothing connected to `images` or `video`, the request body is byte-for-byte what it was
before: `content` stays a plain string, so providers that dislike block arrays keep working.

Video is sent as a `video_url` block. **This is not part of the OpenAI spec** — providers that
accept video at all use this shape, and everything else rejects the request with a readable error.
Mammouth's release notes mention video-to-text with Gemini up to 20 MB, hence the default
`video_max_mb`; the size is checked locally so you get an error immediately instead of after a
long upload.

### Which of your models accept images or video?

The `/models` endpoint doesn't report modalities, and provider docs are usually vague, so ask the
endpoint directly. `tools/probe_modalities.py` sends a 2×2 pixel image (and a tiny generated video
clip) to each model and reports which ones accept it:

```bash
python tools/probe_modalities.py --api-key sk-your-key
```

Useful flags: `--only gpt,gemini` to probe just some models, `--modalities image` to skip video,
`--json report.json` to save the table. It reads `OPENAI_COMPATIBLE_API_KEY` if you omit
`--api-key`. It makes a couple of small requests per model, so it costs a few tokens.

How it avoids lying to you:

- **A text-only baseline runs first.** If a model can't even answer "reply ok", its media verdict
  is reported as `unknown` instead of `no`.
- **The probe image is generated, not hardcoded.** A 2×2 or subtly malformed image gets rejected as
  a bad *image*, which looks exactly like a missing *capability*. It sends a real 256×256 PNG with
  the word HELLO and shows you what the model replied — if it answers "HELLO", it genuinely saw it.
- **Transient failures are retried.** `429`, `5xx` and "no deployments available" are not verdicts;
  after retries they're reported as `unknown`.
- **`no` requires the provider to say so** — the error must mention image/video/modality wording.
  Anything else is `error`, with the full message printed.

Measured on Mammouth (August 2026):

| Model | Text | Image | Video |
|---|---|---|---|
| `gpt-5.6-luna` | yes | **yes** — read "HELLO" off a 256×256 PNG | not tested |
| `gemini-3.6-flash` | yes | **yes** | **yes** — described a white square moving left→right |
| `deepseek-v4-flash-0731` | yes | **no** — `404 "No endpoints found that support image input"` | not tested |

So on this provider it's per-model, and the error for a text-only model is explicit and readable.
Don't generalise from three models — run the probe for the ones you actually use.

## Reusing the last answer

Switch `reuse_last_result` on and the node stops calling the API: it outputs whatever it produced
last time, even if the prompts, model or system message changed. Switch it off and the next run
generates again. Useful for iterating on the rest of a workflow without paying for tokens or
waiting on the model, and for keeping an answer you liked while you change something downstream.

Details:

- The answer is kept on disk, in `user/openai_compatible/last_results.json`, so it survives a
  ComfyUI restart.
- It is stored per node **and** per workflow (keyed by the workflow's uuid plus the node id), so
  two copies of the node — or the same workflow opened twice — never hand each other their text.
- If the toggle is on but nothing has been stored yet, one answer is generated and kept, rather
  than failing.
- While the toggle is on, `model` and `base_url` are not even looked at, so a stale model
  selection or an empty key won't stop the workflow.
- Turning it off does not erase anything: the next successful generation overwrites the stored
  answer.

## API keys

The key is stored in the workflow JSON, so a shared workflow leaks it. To avoid that, leave
`api_key` empty and set an environment variable instead:

- `OPENAI_COMPATIBLE_API_KEY`, or
- `OPENAI_API_KEY`

Or point at a specific variable by typing `env:MY_VARIABLE` into the widget.

## How the model list works

The frontend extension (`web/openai_compatible.js`) posts the URL and key to the
`/openai_compatible/models` route added by this package; ComfyUI's Python process calls
`GET {base_url}/models` and returns the ids. This avoids browser CORS restrictions, and the key
never leaves your machine except towards your endpoint. Results are cached for 30 s; the
**Refresh models** button always bypasses that cache.

## Troubleshooting

- **"Could not load models"** — check the URL ends in `/v1` (or whatever your provider uses) and
  that the key is valid. The exact HTTP status and body are in the toast and the ComfyUI console.
- **Model list is empty after loading a workflow** — the list saved in the workflow is shown first,
  then refreshed; press **Refresh models** if the endpoint was unreachable.
- **`404` on `/chat/completions`** — some providers use a different path prefix; put the full root
  in `base_url`.
- **Provider rejects `temperature`** — set it to `-1`.
- **The node keeps returning the same text** — `reuse_last_result` is on. If it's off, ComfyUI is
  reusing its own cached output because nothing upstream changed; bump `seed`.
- **`400` mentioning image/content type after connecting an image** — that model is text-only. Run
  `tools/probe_modalities.py` to see which of your models take images.
- **Image request is huge or times out** — set `image_max_side` to e.g. `1536`, keep
  `image_format = jpeg`, and use `image_detail = low` if you only need the gist.

## Files

```
comfyui-openai-compatible/
  __init__.py                     extension registration + WEB_DIRECTORY
  nodes.py                        the node
  client.py                       async HTTP client for /models and /chat/completions
  routes.py                       POST /openai_compatible/models
  store.py                        on-disk memory of each node's last answer
  media.py                        IMAGE / VIDEO -> data URLs and content blocks
  tools/probe_modalities.py       asks your endpoint which models accept images/video
  web/openai_compatible.js        model dropdown + refresh button
```
