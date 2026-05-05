"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, TrendingUp, BarChart3, Newspaper, FlaskConical, Radio } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const navItems = [
  { icon: LayoutDashboard, label: "Dashboard", href: "/" },
  { icon: Radio, label: "Live Feed", href: "/live" },
  { icon: TrendingUp, label: "Trend Analysis", href: "/trends" },
  { icon: BarChart3, label: "Forecasting", href: "/forecast" },
  { icon: Newspaper, label: "Event Explorer", href: "/events" },
  { icon: FlaskConical, label: "Ablation", href: "/ablation" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-64 glass border-r border-border/50 transition-transform sm:translate-x-0">
      <div className="flex h-full flex-col px-3 py-4">
        {/* Logo */}
        <div className="mb-10 flex items-center px-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary shadow-lg shadow-primary/20">
            <TrendingUp className="h-6 w-6 text-primary-foreground" />
          </div>
          <span className="ml-3 text-xl font-bold gradient-text">DynamicTrend</span>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center rounded-xl px-4 py-3 text-sm font-medium transition-all duration-200 group",
                  isActive
                    ? "bg-primary/10 text-primary shadow-[inset_0_0_0_1px_rgba(56,189,248,0.2)]"
                    : "text-muted hover:bg-secondary/50 hover:text-foreground"
                )}
              >
                <item.icon className={cn(
                  "mr-3 h-5 w-5 transition-colors",
                  isActive ? "text-primary" : "text-muted group-hover:text-foreground"
                )} />
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="mt-auto space-y-2 px-2 pb-2">
          <Link
            href="https://github.com"
            className="flex items-center rounded-lg px-2 py-2 text-xs font-medium text-muted hover:text-foreground transition-colors"
          >
            {/* <Github className="mr-2 h-4 w-4" /> */}
            Documentation
          </Link>
          <div className="rounded-xl bg-secondary/30 p-4 border border-border/20">
            <p className="text-[10px] uppercase tracking-wider text-muted font-bold mb-1">Status</p>
            <div className="flex items-center">
              <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse mr-2" />
              <p className="text-xs text-foreground font-medium">System Online</p>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
