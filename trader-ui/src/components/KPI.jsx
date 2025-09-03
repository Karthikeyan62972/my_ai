export default function KPI({ label, value, sub, trend, icon, color = "primary" }) {
  const getColorClasses = (color) => {
    const colors = {
      primary: "from-blue-500 to-blue-600",
      success: "from-green-500 to-green-600", 
      warning: "from-yellow-500 to-yellow-600",
      danger: "from-red-500 to-red-600",
      accent: "from-cyan-500 to-cyan-600"
    };
    return colors[color] || colors.primary;
  };

  const getIcon = (icon) => {
    const icons = {
      price: "💰",
      volume: "📊",
      trend: "📈",
      rate: "🎯",
      profit: "💵",
      loss: "📉"
    };
    return icons[icon] || "📊";
  };

  return (
    <div className="kpi group hover:scale-105 transition-all duration-300 cursor-pointer">
      <div className="flex items-center justify-between w-full mb-3">
        <div className="text-sm font-medium text-slate-400 uppercase tracking-wider">{label}</div>
        <div className="text-2xl opacity-80 group-hover:opacity-100 transition-opacity">
          {getIcon(icon)}
        </div>
      </div>
      
      <div className="text-3xl font-bold bg-gradient-to-r bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-300 leading-tight mb-2">
        {value}
      </div>
      
      {sub && (
        <div className="text-sm text-slate-500 font-medium">{sub}</div>
      )}
      
      {trend && (
        <div className={`flex items-center mt-2 text-sm ${
          trend > 0 ? 'text-success' : 'text-danger'
        }`}>
          <span className={`mr-1 ${trend > 0 ? 'rotate-0' : 'rotate-180'}`}>
            {trend > 0 ? '↗' : '↘'}
          </span>
          {Math.abs(trend)}%
        </div>
      )}
      
      <div className={`absolute inset-0 bg-gradient-to-br ${getColorClasses(color)} opacity-0 group-hover:opacity-5 rounded-2xl transition-opacity duration-300`}></div>
    </div>
  );
}

