export default function LoadingSpinner({ size = "md", text = "Loading..." }) {
  const sizeClasses = {
    sm: "w-4 h-4",
    md: "w-8 h-8",
    lg: "w-12 h-12"
  };

  return (
    <div className="flex flex-col items-center justify-center space-y-3">
      <div className={`${sizeClasses[size]} animate-spin rounded-full border-2 border-slate-600 border-t-cyan-500`}></div>
      {text && <div className="text-sm text-slate-400">{text}</div>}
    </div>
  );
}
