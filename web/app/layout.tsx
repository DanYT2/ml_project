import type { Metadata } from "next";
import { Outfit, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  subsets: ["latin"],
  variable: "--font-outfit",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "ASL Interpreter | Real-time Sign Language Recognition",
  description:
    "Convert American Sign Language gestures to text in real-time using AI. Point your camera at sign language and see it translated instantly.",
  keywords: [
    "ASL",
    "sign language",
    "machine learning",
    "accessibility",
    "real-time translation",
  ],
  authors: [{ name: "ML Project Team" }],
  openGraph: {
    title: "ASL Interpreter",
    description: "Real-time American Sign Language to text conversion",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${outfit.variable} ${jetbrainsMono.variable}`}>
      <body className="min-h-screen bg-surface-950 font-sans antialiased">
        {/* Background gradient effects */}
        <div className="fixed inset-0 -z-10">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary-500/10 rounded-full blur-[128px]" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-primary-600/5 rounded-full blur-[128px]" />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_0%,rgb(10,10,12)_70%)]" />
        </div>
        
        {children}
      </body>
    </html>
  );
}



