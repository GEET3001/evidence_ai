import type { ReactNode } from "react";

/**
 * A results section, labelled in the margin rather than with a heading above.
 * The label is the section's identity and stays out of the reading column, so
 * the content — claim text, passages, counts — is the only thing in the flow.
 */
export function Section({
  label,
  meta,
  children,
}: {
  label: string;
  meta?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="border-t border-rule pt-6">
      <div className="grid gap-x-8 gap-y-3 md:grid-cols-[9rem_1fr]">
        <div className="md:sticky md:top-24 md:self-start">
          <h2 className="font-mono text-[11px] font-semibold tracking-[0.14em] text-ink uppercase">
            {label}
          </h2>
          {meta ? (
            <div className="mt-1 font-mono text-[11px] leading-relaxed text-faint">
              {meta}
            </div>
          ) : null}
        </div>
        <div className="min-w-0">{children}</div>
      </div>
    </section>
  );
}
