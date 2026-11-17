import 'dotenv/config';
import express from 'express';
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PYTHON_MASTER_SYSTEM_INSTRUCTION = `You are Python Master AI, an expert Python programmer, debugger, and tutor.
Your goal is to help users write, understand, debug, and optimize Python code.

# Language
Reply in the language the user writes in (English, Urdu, or Roman Urdu).
Keep code, variable names, and comments in English.

# How to respond
- Match the response size to the request. Simple question -> short answer with a small example. Full program or project -> use the structure below.
- Structure for programs/projects:
  1. One or two lines on the approach (and any assumptions you made)
  2. Complete, runnable code (for multi-file /project requests, clearly label each file path above its fenced code block, e.g. \`\`\`python title="app/main.py"\`\`\` or a bold file header like **\`app/main.py\`**)
  3. Short explanation of the key parts
  4. Required libraries and install command (only if non-standard)
  5. How to run it
  6. Expected output
- Ask a clarifying question only when the answer would change the solution significantly. Otherwise state your assumption and proceed.

# Code rules
- Target Python 3.10+. Follow PEP 8, use meaningful names, add comments where logic is not obvious, add type hints for non-trivial functions.
- Code must be complete and runnable. For secrets, read from environment variables (os.getenv); never hardcode keys or passwords.
- Prefer simple, beginner-friendly solutions first; mention advanced options after.
- Avoid eval/exec on user input, use parameterized SQL queries, and validate user input and file paths.
- For ML code: split data before preprocessing, fix random seeds, avoid data leakage, and use metrics that fit the problem.

# Honesty about execution
- Only say code "ran" or "passed" if you actually executed it with a tool.
- Otherwise label output as "Expected output" and say it was not executed.
- When debugging, explain the cause of the error first, then give corrected code.

# Safety
Do not write malware, credential stealers, or tools designed to harm or gain unauthorized access. Offer a safe, legitimate alternative instead.

# Commands
/create   - write a new program
/debug    - find and fix errors, explain the cause
/explain  - explain given code line by line, in simple language
/project  - build a multi-file project with folder structure
/learn    - teach a topic step by step; ask the learner's level first
/optimize - improve speed/readability, explain each change
/quiz     - 5 questions (mix of MCQ and short code tasks) with answers at the end
Without a command, infer the intent from the message.

# Scope
Focus on Python and related tooling (pip, Flask, Django, pandas, ML libraries, SQL, APIs). For unrelated questions, give a brief answer or politely redirect.

# Tone
Friendly, patient, and practical. Encourage beginners; be precise with experts.`;

const PRIMARY_MODEL = 'gemini-4-hydrogen';
const FALLBACK_MODEL = 'gemini-3.1-flash-lite';
const MAX_RETRIES = 3;
const BASE_RETRY_DELAY_MS = 600;
const TRANSIENT_HTTP_STATUSES = new Set([429, 500, 502, 503, 504]);
const EXEC_TIMEOUT_MS = 5000;
const MAX_EXEC_OUTPUT_BYTES = 64 * 1024;
const MAX_EXEC_CODE_BYTES = 64 * 1024;

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not configured on the server. Add it in Settings > Secrets.');
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function extractStatusCode(err: unknown): number | null {
  if (err && typeof err === 'object') {
    const obj = err as { status?: unknown; statusCode?: unknown; code?: unknown; cause?: unknown };
    for (const candidate of [obj.status, obj.statusCode, obj.code]) {
      if (typeof candidate === 'number' && candidate >= 400 && candidate < 600) {
        return candidate;
      }
      if (typeof candidate === 'string') {
        const parsed = Number(candidate);
        if (Number.isInteger(parsed) && parsed >= 400 && parsed < 600) {
          return parsed;
        }
      }
    }
    if (obj.cause) {
      const nested = extractStatusCode(obj.cause);
      if (nested !== null) return nested;
    }
  }

  const raw = err instanceof Error ? err.message : String(err);
  const upper = raw.toUpperCase();

  if (/\b503\b/.test(raw) || upper.includes('UNAVAILABLE') || upper.includes('OVERLOADED')) {
    return 503;
  }
  if (/\b429\b/.test(raw) || upper.includes('RESOURCE_EXHAUSTED') || upper.includes('RATE LIMIT')) {
    return 429;
  }
  if (/\b504\b/.test(raw) || upper.includes('DEADLINE_EXCEEDED') || upper.includes('TIMED OUT')) {
    return 504;
  }
  if (/\b502\b/.test(raw) || upper.includes('BAD GATEWAY')) {
    return 502;
  }
  if (/\b500\b/.test(raw) || upper.includes('INTERNAL')) {
    return 500;
  }
  if (/\b403\b/.test(raw) || upper.includes('PERMISSION_DENIED')) {
    return 403;
  }
  if (/\b401\b/.test(raw) || upper.includes('UNAUTHENTICATED')) {
    return 401;
  }
  if (/\b400\b/.test(raw) || upper.includes('API_KEY_INVALID') || upper.includes('INVALID_ARGUMENT')) {
    return 400;
  }
  if (/\b404\b/.test(raw) || upper.includes('NOT_FOUND')) {
    return 404;
  }
  return null;
}

function isRetryableError(err: unknown): boolean {
  const status = extractStatusCode(err);
  if (status !== null && TRANSIENT_HTTP_STATUSES.has(status)) {
    return true;
  }
  const raw = (err instanceof Error ? err.message : String(err)).toUpperCase();
  return (
    raw.includes('ECONNRESET') ||
    raw.includes('ETIMEDOUT') ||
    raw.includes('EAI_AGAIN') ||
    raw.includes('ENOTFOUND') ||
    raw.includes('FETCH FAILED') ||
    raw.includes('SOCKET HANG UP') ||
    raw.includes('NETWORK')
  );
}

function formatErrorMessage(err: unknown): {
  status: number;
  message: string;
  retryable: boolean;
} {
  const raw = err instanceof Error ? err.message : String(err);
  if (raw.includes('GEMINI_API_KEY')) {
    return {
      status: 500,
      message: 'The AI service is not configured yet. Please check Settings > Secrets.',
      retryable: false,
    };
  }

  const detectedStatus = extractStatusCode(err);

  if (detectedStatus === 503) {
    return {
      status: 503,
      message:
        'The AI service is temporarily unavailable due to high demand. Please wait a moment and click Retry.',
      retryable: true,
    };
  }
  if (detectedStatus === 429) {
    return {
      status: 429,
      message:
        'The AI service is receiving too many requests right now. Please wait a moment and click Retry.',
      retryable: true,
    };
  }
  if (detectedStatus === 502 || detectedStatus === 504) {
    return {
      status: detectedStatus,
      message:
        'The connection to the AI service timed out. Please click Retry to try again.',
      retryable: true,
    };
  }
  if (detectedStatus === 500) {
    return {
      status: 500,
      message:
        'The AI service encountered a temporary server error. Please click Retry to try again.',
      retryable: true,
    };
  }
  if (detectedStatus === 400 || detectedStatus === 401 || detectedStatus === 403) {
    return {
      status: detectedStatus,
      message:
        'API key rejected or invalid. Please verify your Gemini API key in Settings > Secrets.',
      retryable: false,
    };
  }
  if (detectedStatus === 404) {
    return {
      status: 404,
      message: 'The requested AI model is currently unavailable.',
      retryable: false,
    };
  }

  if (isRetryableError(err)) {
    return {
      status: 503,
      message:
        'The AI service is temporarily unavailable. Please wait a moment and click Retry.',
      retryable: true,
    };
  }

  // Never pass raw JSON payloads to the client
  const trimmed = raw.trim();
  const looksLikeJson =
    trimmed.startsWith('{') ||
    trimmed.startsWith('[') ||
    /"error"\s*:|"code"\s*:\s*\d{3}/.test(trimmed);

  return {
    status: 500,
    message:
      !trimmed || looksLikeJson
        ? 'The AI service could not complete your request right now. Please click Retry.'
        : trimmed,
    retryable: true,
  };
}

function sleepAbortable(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) {
    return Promise.reject(new Error('Client disconnected'));
  }
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new Error('Client disconnected'));
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

async function withRetry<T>(
  operation: (model: string, attemptIndex: number) => Promise<T>,
  options: {
    maxRetries?: number;
    baseDelayMs?: number;
    signal?: AbortSignal;
    shouldRetry?: (err: unknown) => boolean;
    onRetry?: (attempt: number, delayMs: number, nextModel: string, err: unknown) => void;
  } = {}
): Promise<T> {
  const maxRetries = options.maxRetries ?? MAX_RETRIES;
  const baseDelayMs = options.baseDelayMs ?? BASE_RETRY_DELAY_MS;

  let attempt = 0;
  while (true) {
    if (options.signal?.aborted) {
      throw new Error('Client disconnected');
    }
    // Use primary model for initial attempt and first retry, then fall back to flash-lite on persistent transient spikes
    const activeModel = attempt >= 1 ? FALLBACK_MODEL : PRIMARY_MODEL;
    try {
      return await operation(activeModel, attempt);
    } catch (err) {
      if (options.signal?.aborted) {
        throw err;
      }
      const allowedByCaller = options.shouldRetry ? options.shouldRetry(err) : true;
      if (!allowedByCaller || !isRetryableError(err) || attempt >= maxRetries) {
        throw err;
      }
      attempt += 1;
      const nextModel = attempt >= 1 ? FALLBACK_MODEL : PRIMARY_MODEL;
      const exponentialDelay = baseDelayMs * Math.pow(2, attempt - 1);
      const jitter = Math.floor(Math.random() * 200);
      const totalDelay = exponentialDelay + jitter;
      options.onRetry?.(attempt, totalDelay, nextModel, err);
      await sleepAbortable(totalDelay, options.signal);
    }
  }
}

function summarizePythonError(stderr: string, timedOut: boolean): string | undefined {
  if (timedOut) {
    return 'Execution timed out after 5 seconds. Check for infinite loops or blocking input() calls.';
  }
  const trimmed = stderr.trim();
  if (!trimmed) return undefined;

  const lines = trimmed.split('\n').map((l) => l.trim()).filter(Boolean);
  const lastLine = lines[lines.length - 1] || '';

  if (lastLine.startsWith('EOFError')) {
    return 'EOFError: Your script called input(), but no matching line was provided in Program Input (stdin).';
  }
  if (lastLine.startsWith('SyntaxError') || lastLine.startsWith('IndentationError')) {
    return `${lastLine} — Check syntax, colons, quotes, or indentation.`;
  }
  if (lastLine) {
    return lastLine;
  }
  return 'Python exited with an error.';
}

async function executeIsolatedPython(code: string, stdinInput: string): Promise<{
  stdout: string;
  stderr: string;
  exitCode: number | null;
  timedOut: boolean;
  durationMs: number;
  pythonVersion: string;
  errorSummary?: string;
}> {
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'py-sandbox-'));
  const userScriptPath = path.join(tempDir, 'main.py');
  const runnerPath = path.join(tempDir, '__runner__.py');

  const runnerSource = [
    'import resource, sys, traceback',
    'try:',
    '    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))',
    '    resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))',
    '    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))',
    'except Exception:',
    '    pass',
    'with open("main.py", "r", encoding="utf-8") as f:',
    '    source = f.read()',
    'try:',
    '    compiled = compile(source, "main.py", "exec")',
    '    exec_globals = {"__name__": "__main__", "__file__": "main.py"}',
    '    exec(compiled, exec_globals)',
    'except SystemExit as exc:',
    '    code = exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)',
    '    if exc.code and not isinstance(exc.code, int):',
    '        print(exc.code, file=sys.stderr)',
    '    sys.exit(code)',
    'except Exception:',
    '    tb_lines = traceback.format_exc().splitlines()',
    '    filtered = [line for line in tb_lines if "__runner__.py" not in line]',
    '    print("\\n".join(filtered), file=sys.stderr)',
    '    sys.exit(1)',
  ].join('\n');

  const startedAt = Date.now();

  try {
    await fs.writeFile(userScriptPath, code, 'utf8');
    await fs.writeFile(runnerPath, runnerSource, 'utf8');

    return await new Promise((resolve) => {
      const child = spawn('python3', ['-I', runnerPath], {
        cwd: tempDir,
        env: {
          PATH: '/usr/local/bin:/usr/bin:/bin',
          HOME: tempDir,
          TMPDIR: tempDir,
          PYTHONDONTWRITEBYTECODE: '1',
          PYTHONUNBUFFERED: '1',
          PYTHONIOENCODING: 'utf-8',
        },
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';
      let totalBytes = 0;
      let truncated = false;
      let timedOut = false;
      let settled = false;

      const finish = (exitCode: number | null) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);

        const cleanStderr = stderr.replaceAll(tempDir, '/workspace');
        const finalStderr = truncated
          ? `${cleanStderr}\n[Execution stopped: output exceeded 64 KB limit.]`.trim()
          : cleanStderr;

        resolve({
          stdout,
          stderr: finalStderr,
          exitCode: truncated && exitCode === null ? 1 : exitCode,
          timedOut,
          durationMs: Date.now() - startedAt,
          pythonVersion: '3.11',
          errorSummary:
            truncated
              ? 'Output exceeded the 64 KB console limit and execution was stopped.'
              : summarizePythonError(finalStderr, timedOut),
        });
      };

      const timer = setTimeout(() => {
        timedOut = true;
        child.kill('SIGKILL');
      }, EXEC_TIMEOUT_MS);

      child.stdout.on('data', (chunk: Buffer) => {
        if (truncated) return;
        const remaining = MAX_EXEC_OUTPUT_BYTES - totalBytes;
        if (chunk.length > remaining) {
          stdout += chunk.subarray(0, Math.max(0, remaining)).toString('utf8');
          truncated = true;
          child.kill('SIGKILL');
          return;
        }
        totalBytes += chunk.length;
        stdout += chunk.toString('utf8');
      });

      child.stderr.on('data', (chunk: Buffer) => {
        if (truncated) return;
        const remaining = MAX_EXEC_OUTPUT_BYTES - totalBytes;
        if (chunk.length > remaining) {
          stderr += chunk.subarray(0, Math.max(0, remaining)).toString('utf8');
          truncated = true;
          child.kill('SIGKILL');
          return;
        }
        totalBytes += chunk.length;
        stderr += chunk.toString('utf8');
      });

      child.on('error', (spawnErr) => {
        stderr += `Failed to launch Python runtime: ${spawnErr.message}`;
        finish(1);
      });

      child.on('close', (codeVal) => {
        finish(codeVal);
      });

      if (stdinInput) {
        child.stdin.write(stdinInput);
      }
      child.stdin.end();
    });
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true }).catch(() => {});
  }
}

interface IncomingMessage {
  role: 'user' | 'model';
  text: string;
  attachedCode?: string;
  attachedTraceback?: string;
}

interface ChatRequestBody {
  messages?: IncomingMessage[];
  languagePreference?: string;
  skillLevel?: string;
}

interface QuizRequestBody {
  topic?: string;
  skillLevel?: string;
  languagePreference?: string;
}

function buildDynamicSystemInstruction(languagePreference?: string, skillLevel?: string): string {
  const extras: string[] = [];

  if (languagePreference && languagePreference !== 'auto') {
    const langLabels: Record<string, string> = {
      english: 'English',
      urdu: 'Urdu (اردو script)',
      'roman-urdu': 'Roman Urdu (Urdu written in Latin alphabet)',
    };
    const label = langLabels[languagePreference];
    if (label) {
      extras.push(
        `User language preference override: Reply in ${label} while keeping all Python code, variable names, and code comments in English.`
      );
    }
  }

  if (skillLevel && skillLevel !== 'auto') {
    extras.push(
      `User declared skill level: ${skillLevel}. Tailor explanations and depth appropriately while keeping code clean and PEP 8 compliant.`
    );
  }

  if (extras.length === 0) {
    return PYTHON_MASTER_SYSTEM_INSTRUCTION;
  }

  return `${PYTHON_MASTER_SYSTEM_INSTRUCTION}\n\n# Session Preferences\n${extras.join('\n')}`;
}

async function createServer() {
  const app = express();
  app.use(express.json({ limit: '2mb' }));

  app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', service: 'python-master-ai' });
  });

  app.post('/api/chat', async (req, res) => {
    const body = (req.body || {}) as ChatRequestBody;
    const messages = Array.isArray(body.messages) ? body.messages : [];

    if (messages.length === 0) {
      res.status(400).json({ error: 'At least one message is required.' });
      return;
    }

    let ai: GoogleGenAI;
    try {
      ai = getGeminiClient();
    } catch (err) {
      const formatted = formatErrorMessage(err);
      res.status(formatted.status).json({ error: formatted.message, retryable: formatted.retryable });
      return;
    }

    const abortController = new AbortController();
    res.on('close', () => {
      if (!res.writableEnded) {
        abortController.abort();
      }
    });

    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.setHeader('x-ai-retried', 'true');
    res.flushHeaders();
    res.write(': connected\n\n');

    const contents = messages.map((msg) => {
      let fullText = msg.text;
      if (msg.attachedCode && msg.attachedCode.trim()) {
        fullText += `\n\nAttached Python code:\n\`\`\`python\n${msg.attachedCode.trim()}\n\`\`\``;
      }
      if (msg.attachedTraceback && msg.attachedTraceback.trim()) {
        fullText += `\n\nError / Traceback:\n\`\`\`text\n${msg.attachedTraceback.trim()}\n\`\`\``;
      }
      return {
        role: msg.role,
        parts: [{ text: fullText }],
      };
    });

    const systemInstruction = buildDynamicSystemInstruction(
      body.languagePreference,
      body.skillLevel
    );

    let emittedAnyChunk = false;

    try {
      await withRetry(
        async (modelName) => {
          const stream = await ai.models.generateContentStream({
            model: modelName,
            contents,
            config: {
              systemInstruction,
            },
          });

          for await (const chunk of stream) {
            if (abortController.signal.aborted) {
              return;
            }
            const text = chunk.text;
            if (text) {
              emittedAnyChunk = true;
              res.write(`data: ${JSON.stringify({ text })}\n\n`);
            }
          }
        },
        {
          maxRetries: MAX_RETRIES,
          baseDelayMs: BASE_RETRY_DELAY_MS,
          signal: abortController.signal,
          shouldRetry: () => !emittedAnyChunk,
          onRetry: (attempt) => {
            if (!res.writableEnded && !emittedAnyChunk) {
              res.write(
                `data: ${JSON.stringify({
                  retrying: true,
                  attempt,
                  maxRetries: MAX_RETRIES,
                })}\n\n`
              );
            }
          },
        }
      );

      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
        res.end();
      }
    } catch (err) {
      if (abortController.signal.aborted || res.writableEnded) {
        return;
      }
      const formatted = formatErrorMessage(err);
      res.write(
        `data: ${JSON.stringify({
          error: formatted.message,
          status: formatted.status,
          retryable: formatted.retryable,
        })}\n\n`
      );
      res.end();
    }
  });

  app.post('/api/execute', async (req, res) => {
    const body = (req.body || {}) as { code?: unknown; stdin?: unknown };
    const code = typeof body.code === 'string' ? body.code : '';
    const stdinInput = typeof body.stdin === 'string' ? body.stdin : '';

    if (!code.trim()) {
      res.status(400).json({ error: 'Please provide Python code to run.' });
      return;
    }

    if (Buffer.byteLength(code, 'utf8') > MAX_EXEC_CODE_BYTES) {
      res.status(400).json({ error: 'Python script exceeds the 64 KB size limit.' });
      return;
    }

    try {
      const result = await executeIsolatedPython(code, stdinInput);
      res.json(result);
    } catch {
      res.status(500).json({ error: 'Unable to execute Python code in the sandbox right now.' });
    }
  });

  app.post('/api/quiz', async (req, res) => {
    const body = (req.body || {}) as QuizRequestBody;
    const topic = (body.topic || 'Python fundamentals, functions, and data structures').trim();
    const skillLevel =
      body.skillLevel && body.skillLevel !== 'auto' ? body.skillLevel : 'intermediate';
    const languagePreference = body.languagePreference || 'english';

    let ai: GoogleGenAI;
    try {
      ai = getGeminiClient();
    } catch (err) {
      const formatted = formatErrorMessage(err);
      res.status(formatted.status).json({ error: formatted.message, retryable: formatted.retryable });
      return;
    }

    const langInstruction =
      languagePreference === 'urdu'
        ? 'Write question prompts and explanations in Urdu (اردو), keeping Python code snippets in English.'
        : languagePreference === 'roman-urdu'
          ? 'Write question prompts and explanations in Roman Urdu, keeping Python code snippets in English.'
          : 'Write question prompts and explanations in English.';

    const abortController = new AbortController();
    res.on('close', () => {
      if (!res.writableEnded) {
        abortController.abort();
      }
    });

    try {
      const response = await withRetry(
        (modelName) =>
          ai.models.generateContent({
            model: modelName,
            contents: `Generate a 5-question Python 3.10+ quiz on "${topic}" for a ${skillLevel} learner. Include 3 multiple-choice questions ("mcq" with 4 clear options) and 2 short code tasks ("code_task" where the learner writes or predicts a short Python snippet). ${langInstruction} Keep explanations concise and accurate.`,
            config: {
              systemInstruction: PYTHON_MASTER_SYSTEM_INSTRUCTION,
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  topic: { type: Type.STRING },
                  difficulty: { type: Type.STRING },
                  questions: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        id: { type: Type.INTEGER },
                        type: { type: Type.STRING },
                        prompt: { type: Type.STRING },
                        codeSnippet: { type: Type.STRING },
                        options: {
                          type: Type.ARRAY,
                          items: { type: Type.STRING },
                        },
                        correctAnswer: { type: Type.STRING },
                        explanation: { type: Type.STRING },
                      },
                      required: ['id', 'type', 'prompt', 'correctAnswer', 'explanation'],
                    },
                  },
                },
                required: ['topic', 'difficulty', 'questions'],
              },
            },
          }),
        {
          maxRetries: MAX_RETRIES,
          baseDelayMs: BASE_RETRY_DELAY_MS,
          signal: abortController.signal,
        }
      );

      const raw = (response.text ?? '{}').trim();
      const parsed = JSON.parse(raw);
      res.setHeader('x-ai-retried', 'true');
      res.json(parsed);
    } catch (err) {
      if (abortController.signal.aborted || res.writableEnded) {
        return;
      }
      const formatted = formatErrorMessage(err);
      res.setHeader('x-ai-retried', 'true');
      // Avoid 502/503/504 status codes being replaced with Nginx warmup.html while preserving retryable semantics
      const httpStatus =
        formatted.status === 502 || formatted.status === 503 || formatted.status === 504
          ? 500
          : formatted.status;
      res.status(httpStatus).json({
        error: formatted.message,
        statusCode: formatted.status,
        retryable: formatted.retryable,
      });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  const port = Number(process.env.PORT) || 3000;
  app.listen(port, '0.0.0.0', () => {
    console.log(`Python Master AI server listening on http://0.0.0.0:${port}`);
  });
}

createServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
