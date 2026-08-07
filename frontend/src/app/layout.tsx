import type { Metadata } from "next";
import { IBM_Plex_Mono, Instrument_Sans, Source_Serif_4 } from "next/font/google";
import "./globals.css";

const instrumentSans = Instrument_Sans({
  variable: "--font-instrument-sans",
  subsets: ["latin"],
});

const sourceSerif = Source_Serif_4({
  variable: "--font-source-serif",
  subsets: ["latin"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "EvidenceAI",
  description: "Explainable research claim verification for mental health literature.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${instrumentSans.variable} ${sourceSerif.variable} ${plexMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col">
        {/* Sticky so it stays on screen through a long results page, which is
            the view where mistaking this for clinical advice would matter. */}
        <div className="sticky top-0 z-50 border-b border-rule bg-sunk/95 backdrop-blur">
          <p className="mx-auto max-w-5xl px-6 py-2 font-mono text-[11px] leading-relaxed tracking-wide text-muted">
            <span className="font-semibold text-ink">SCOPE</span> — Literature triage
            aid for researchers and students. Not a diagnostic or treatment-advice
            tool.
          </p>
        </div>
        {children}
      </body>
    </html>
  );
}
