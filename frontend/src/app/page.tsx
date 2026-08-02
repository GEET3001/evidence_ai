"use client";

import { useState, type FormEvent } from "react";
import { ApiError, verifyClaim } from "@/lib/api";
import type { VerdictResponse } from "@/types";

export default function Home() {
  const [claim, setClaim] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<VerdictResponse | null>(null);

  const canSubmit = claim.trim().length > 0 && !loading;

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      setResult(await verifyClaim(claim.trim()));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something unexpected went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-6 px-6 py-12">
      <div>
        <h1 className="text-2xl font-semibold text-zinc-900 dark:text-zinc-50">
          Verify a research claim
        </h1>
        <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
          Enter a claim about mental health research and EvidenceAI will retrieve
          and classify relevant evidence from its corpus.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <textarea
          value={claim}
          onChange={(e) => setClaim(e.target.value)}
          placeholder="Mindfulness meditation reduces anxiety in college students"
          rows={4}
          className="w-full resize-none rounded-md border border-zinc-300 p-3 text-sm text-zinc-900 focus:border-zinc-500 focus:outline-none dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50"
        />
        <button
          type="submit"
          disabled={!canSubmit}
          className="self-start rounded-md bg-zinc-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40 dark:bg-zinc-50 dark:text-zinc-900"
        >
          {loading ? "Verifying…" : "Verify"}
        </button>
      </form>

      {loading && (
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Retrieving and classifying evidence — this can take a few seconds.
        </p>
      )}

      {error && (
        <div className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200">
          {error}
        </div>
      )}

      {result && (
        <div className="flex flex-col gap-2">
          <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
            Raw response (temporary — the real results UI is a later task)
          </h2>
          <pre className="max-h-[600px] overflow-auto rounded-md bg-zinc-100 p-3 text-xs text-zinc-800 dark:bg-zinc-900 dark:text-zinc-200">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </main>
  );
}
