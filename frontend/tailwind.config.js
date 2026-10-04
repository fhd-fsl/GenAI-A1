/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      "colors": {
        "on-error": "#690005", "primary-fixed": "#acedff", "secondary-fixed": "#6ffbbe",
        "background": "#111319", "inverse-on-surface": "#2e3037", "on-primary": "#003640",
        "on-secondary": "#003824", "primary-fixed-dim": "#4cd7f6", "inverse-surface": "#e2e2eb",
        "outline-variant": "#3d494c", "surface-container-low": "#191b22", "on-surface-variant": "#bcc9cd",
        "surface-variant": "#33343b", "primary-container": "#06b6d4", "inverse-primary": "#00687a",
        "surface": "#111319", "surface-dim": "#111319", "surface-container-high": "#282a30",
        "on-primary-container": "#00424f", "surface-bright": "#373940", "error-container": "#93000a",
        "on-secondary-container": "#00311f", "tertiary": "#d0bcff", "surface-tint": "#4cd7f6",
        "on-secondary-fixed-variant": "#005236", "secondary-container": "#00a572", "primary": "#4cd7f6",
        "on-surface": "#e2e2eb", "tertiary-container": "#b395ff", "on-tertiary-container": "#4900ae",
        "on-tertiary-fixed-variant": "#5516be", "on-tertiary-fixed": "#23005c", "outline": "#869397",
        "on-tertiary": "#3c0091", "on-primary-fixed": "#001f26", "tertiary-fixed": "#e9ddff",
        "surface-container-lowest": "#0c0e14", "error": "#ffb4ab", "on-background": "#e2e2eb",
        "secondary-fixed-dim": "#4edea3", "on-primary-fixed-variant": "#004e5c", "on-error-container": "#ffdad6",
        "secondary": "#4edea3", "tertiary-fixed-dim": "#d0bcff", "surface-container-highest": "#33343b",
        "surface-container": "#1e1f26", "on-secondary-fixed": "#002113"
      },
      "borderRadius": {
        "DEFAULT": "0.25rem", "lg": "0.5rem", "xl": "0.75rem", "full": "9999px"
      },
      "spacing": {
        "margin": "1.5rem", "gutter-mobile": "0.75rem", "space-xl": "2rem",
        "space-md": "1rem", "margin-mobile": "1rem", "space-sm": "0.5rem",
        "gutter": "1rem", "space-xs": "0.25rem", "space-lg": "1.5rem"
      },
      "fontFamily": {
        "headline-lg-mobile": ["Inter"], "headline-md": ["Inter"], "body-sm": ["Inter"],
        "label-sm": ["Inter"], "label-md": ["Inter"], "display": ["Inter"],
        "body-md": ["Inter"], "body-lg": ["Inter"], "headline-lg": ["Inter"]
      },
      "fontSize": {
        "headline-lg-mobile": ["1.5rem", { "lineHeight": "2rem", "letterSpacing": "-0.015em", "fontWeight": "600" }],
        "headline-md": ["1.25rem", { "lineHeight": "1.75rem", "letterSpacing": "-0.01em", "fontWeight": "600" }],
        "body-sm": ["0.75rem", { "lineHeight": "1.125rem", "letterSpacing": "0.01em", "fontWeight": "400" }],
        "label-sm": ["0.6875rem", { "lineHeight": "0.875rem", "letterSpacing": "0.04em", "fontWeight": "600" }],
        "label-md": ["0.75rem", { "lineHeight": "1rem", "letterSpacing": "0.02em", "fontWeight": "500" }],
        "display": ["3rem", { "lineHeight": "3.5rem", "letterSpacing": "-0.025em", "fontWeight": "700" }],
        "body-md": ["0.875rem", { "lineHeight": "1.375rem", "letterSpacing": "0em", "fontWeight": "400" }],
        "body-lg": ["1rem", { "lineHeight": "1.5rem", "letterSpacing": "0em", "fontWeight": "400" }],
        "headline-lg": ["2rem", { "lineHeight": "2.5rem", "letterSpacing": "-0.02em", "fontWeight": "600" }]
      }
    }
  },
  plugins: [],
}
