/** @type {import('next').NextConfig} */
const nextConfig = {
  // Fully static output. There are no API routes and no runtime data fetching,
  // so the whole site is prerendered at build time from the JSON in results/.
  output: "export",
  images: { unoptimized: true },
  reactStrictMode: true,
};

export default nextConfig;
