import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Mirage",
  description:
    "A backtest overfitting auditor. Estimates the probability that a trading strategy result is a false discovery rather than a real edge.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
