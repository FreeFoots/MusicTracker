import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "MusicTracker · 音乐数据工作台",
  description: "一句话，发现歌曲的跨平台数据与演唱版本。",
};
export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
