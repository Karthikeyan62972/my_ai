import { useState } from "react";

export default function Sidebar({ currentPage, onPageChange, isOpen, onToggle }) {
  const menuItems = [
    { id: "dashboard", label: "Dashboard", icon: "📊", color: "from-blue-500 to-blue-600" },
    { id: "overview", label: "Market Overview", icon: "🌍", color: "from-green-500 to-green-600" },
    { id: "analytics", label: "Analytics", icon: "📈", color: "from-purple-500 to-purple-600" },
    { id: "settings", label: "Settings", icon: "⚙️", color: "from-gray-500 to-gray-600" },
  ];

  return (
    <>
      {/* Sidebar */}
      <div className={`fixed left-0 top-0 h-full w-64 bg-gradient-to-b from-slate-800/95 to-slate-900/95 backdrop-blur-xl border-r border-slate-700/30 transition-all duration-300 z-50 ${
        isOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        {/* Logo */}
        <div className="p-6 border-b border-slate-700/30">
          <div className="flex items-center space-x-3">
            <div className="text-3xl">🚀</div>
            <div>
              <div className="text-lg font-bold text-white">ML Trader</div>
              <div className="text-xs text-slate-400">Pro Dashboard</div>
            </div>
          </div>
        </div>

        {/* Navigation Menu */}
        <nav className="p-4 space-y-2">
          {menuItems.map((item) => (
            <button
              key={item.id}
              onClick={() => onPageChange(item.id)}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-200 group ${
                currentPage === item.id
                  ? `bg-gradient-to-r ${item.color} text-white shadow-lg`
                  : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
              }`}
            >
              <div className="text-xl">{item.icon}</div>
              <span className="font-medium">{item.label}</span>
              {currentPage === item.id && (
                <div className="ml-auto w-2 h-2 bg-white rounded-full animate-pulse"></div>
              )}
            </button>
          ))}
        </nav>

        {/* Bottom Section */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-slate-700/30">
          <div className="text-center">
            <div className="text-xs text-slate-500 mb-2">Trading Status</div>
            <div className="flex items-center justify-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm text-green-400 font-medium">Active</span>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
          onClick={onToggle}
        />
      )}

      {/* Mobile Toggle Button */}
      <button
        onClick={onToggle}
        className="fixed top-4 left-4 z-50 lg:hidden p-2 bg-slate-800/80 backdrop-blur-sm rounded-lg border border-slate-700/30 hover:bg-slate-700/80 transition-colors"
      >
        <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>
    </>
  );
}
