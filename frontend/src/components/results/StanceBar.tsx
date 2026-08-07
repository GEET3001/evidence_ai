/**
 * The support / contradict / neutral split as one proportional bar.
 *
 * Neutral is shown because leaving it out would make three neutral passages and
 * no directional ones look like a full bar of agreement.
 */
export function StanceBar({
  support,
  contradict,
  neutral,
  tone = "light",
}: {
  support: number;
  contradict: number;
  neutral: number;
  tone?: "light" | "dark";
}) {
  const total = support + contradict + neutral;
  if (total === 0) return null;

  const parts = [
    { key: "support", count: support, className: "bg-support" },
    { key: "contradict", count: contradict, className: "bg-contradict" },
    { key: "neutral", count: neutral, className: "bg-neutral" },
  ].filter((part) => part.count > 0);

  return (
    <div
      className={`flex h-2 w-full overflow-hidden rounded-full ${
        tone === "dark" ? "bg-banner-rule" : "bg-sunk"
      }`}
      role="img"
      aria-label={`${support} supporting, ${contradict} contradicting, ${neutral} neutral of ${total} passages`}
    >
      {parts.map((part) => (
        <span
          key={part.key}
          className={part.className}
          style={{ width: `${(part.count / total) * 100}%` }}
        />
      ))}
    </div>
  );
}
