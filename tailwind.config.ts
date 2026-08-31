import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0a0c10",
          900: "#0f1218",
          800: "#161a23",
          700: "#1f242f",
          600: "#2b313d",
        },
        mist: {
          100: "#e8eaef",
          300: "#a8aebc",
          500: "#767d8d",
          700: "#4a5162",
        },
        signal: "#e0a458",
      },
      fontFamily: {
        sans: ["var(--font-sans)"],
        mono: ["var(--font-mono)"],
      },
      maxWidth: {
        prose: "68ch",
      },
    },
  },
  plugins: [],
};

export default config;
