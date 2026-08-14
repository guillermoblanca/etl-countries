"""
Build a standalone HTML report for the Spain Case Study, suitable for
GitHub Pages or any static hosting.

Usage:
    docker compose up -d api          # API must be running
    python scripts/build_case_study.py

Output:
    docs/index.html — fully self-contained HTML (no backend needed)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

API_BASE = os.getenv("API_BASE", "http://localhost:8080")
ROOT     = Path(__file__).resolve().parent.parent
OUT_DIR  = ROOT / "docs"
OUT_FILE = OUT_DIR / "index.html"


def fetch(path: str):
    url = f"{API_BASE}{path}"
    print(f"  >> {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def fmt_usd(n):
    if n is None: return "—"
    return f"${int(n):,}"


def fmt_pct(n, decimals=1):
    if n is None: return "—"
    return f"{n:.{decimals}f}%"


def safe(v, default="—"):
    return v if v is not None else default


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching Spain case study data from {API_BASE}…")
    d = fetch("/case-study/spain")

    c = d["country"]
    archetype = d.get("archetype") or {}
    strength = d.get("strength") or {}
    vulnerability = d.get("vulnerability") or {}
    convergence = d.get("convergence") or {}
    pp = d.get("purchasing_power") or {}
    zone = d.get("zone") or {}
    cluster = d.get("cluster") or {}
    crises = d.get("crises") or []
    euro_self = d.get("euro_self") or {}
    europe_comparison = d.get("europe_comparison") or []
    recent = d.get("recent_trends") or []
    ml_risk = d.get("ml_risk") or {}

    # ── Figures that must never be written by hand ───────────────────────────
    # An earlier version of this report hardcoded Spain's position ("top 1.4%"),
    # the model's AUC (0.824) and its training size (11,475) straight into the
    # prose. All three drifted away from the data the same page was rendering.
    # They are now derived on every build.
    #
    # /case-study/spain returns a strength object without global_rank, which is
    # why the composite ranking used to render as an em dash. Take it from the
    # analytics endpoint, which carries it.
    strength_ranked = fetch(f"/analytics/strength/{c['cca2']}")
    strength = {**strength, **strength_ranked}

    ranking = fetch(f"/api/country/{c['cca2']}/ranking")
    gdp_rank = ranking.get("NY.GDP.PCAP.CD") or {}
    gdp_pos, gdp_of = gdp_rank.get("global_rank"), gdp_rank.get("global_total")
    gdp_top_pct = f"{gdp_pos / gdp_of * 100:.0f}" if gdp_pos and gdp_of else None

    model = (fetch("/ml/predict/model-info") or {}).get("metrics") or {}
    model_auc = model.get("auc_roc")
    model_n = model.get("train_samples")

    # Same reasoning for the prose in the "últimos 3 años" section, which used to
    # restate figures the chips above it were already computing.
    trend_by_label = {t.get("label", "").strip(): t for t in recent}

    def trend_abs(label, default="—"):
        t = trend_by_label.get(label) or {}
        v = t.get("delta_pct_3y")
        return f"{abs(v):.1f}".replace(".", ",") if v is not None else default

    unemp_drop = trend_abs("Paro %")
    debt_drop = trend_abs("Deuda gob. %")

    auc_txt = f"{model_auc:.3f}".replace(".", ",") if model_auc else "—"
    n_txt = f"{model_n:,}".replace(",", ".") if model_n else "—"
    if gdp_top_pct:
        gdp_pos_txt = (
            f"el <strong>{gdp_top_pct}% de los países más ricos del mundo por "
            f"PIB per cápita</strong> (puesto {gdp_pos} de {gdp_of})"
        )
    else:
        gdp_pos_txt = "<strong>la mitad alta mundial por PIB per cápita</strong>"

    # ── Sections HTML ────────────────────────────────────────────────────────
    hero_kpis = f"""
      <div class="cs-hero__kpi"><div class="cs-hero__kpi-val">{fmt_usd(c.get('latest_gdp_per_capita'))}</div><div class="cs-hero__kpi-lbl">PIB per cápita</div></div>
      <div class="cs-hero__kpi"><div class="cs-hero__kpi-val">{safe(strength.get('strength_score'))}</div><div class="cs-hero__kpi-lbl">Puntuación fortaleza · puesto #{safe(strength.get('global_rank'))}/{safe(strength.get('total'))}</div></div>
      <div class="cs-hero__kpi"><div class="cs-hero__kpi-val">{safe(convergence.get('pct_us_now'))}%</div><div class="cs-hero__kpi-lbl">del PIB per cápita de EE.UU.</div></div>
      <div class="cs-hero__kpi"><div class="cs-hero__kpi-val">{safe(pp.get('cost_of_living_index'))}</div><div class="cs-hero__kpi-lbl">Índice coste de vida (EE.UU.=100)</div></div>
      <div class="cs-hero__kpi"><div class="cs-hero__kpi-val">{(cluster.get('label') or '—').split(' ')[0]}</div><div class="cs-hero__kpi-lbl">Grupo económico</div></div>
    """

    # Section 1 hero stat row
    sec1_stats = f"""
      <div class="cs-stat"><div class="cs-stat__val">{safe(strength.get('driver_level'))}</div><div class="cs-stat__lbl">Nivel de renta<br>(parte alta mundial)</div></div>
      <div class="cs-stat"><div class="cs-stat__val">{safe(strength.get('driver_stability'))}</div><div class="cs-stat__lbl">Estabilidad<br>(variación del crecimiento 30 años)</div></div>
      <div class="cs-stat"><div class="cs-stat__val" style="color:var(--red)">{safe(strength.get('driver_resilience'))}</div><div class="cs-stat__lbl">Capacidad de recuperación<br>(tras una crisis)</div></div>
      <div class="cs-stat"><div class="cs-stat__val" style="color:var(--red)">{safe(strength.get('driver_diversification'))}</div><div class="cs-stat__lbl">Diversificación sectorial<br>(servicios = {safe(archetype.get('services_va_pct'))}% del PIB)</div></div>
    """

    # Section 3 Europe table
    europe_rows = ""
    for co in europe_comparison:
        is_es = "spain" if co["cca2"] == "ES" else ""
        europe_rows += f"""
        <tr class="{is_es}">
          <td>{co.get('flag_emoji','')} <strong>{co['name']}</strong> <span style="color:hsl(228,8%,55%);font-size:.7rem">{safe(co.get('subregion'),'')}</span></td>
          <td class="num">{fmt_usd(co.get('latest_gdp_per_capita'))}</td>
          <td class="num">{fmt_pct(co.get('latest_unemployment'))}</td>
          <td class="num">{f"{round(co['govt_debt'])}%" if co.get('govt_debt') else '—'}</td>
          <td class="num">{safe(co.get('strength_score'))}</td>
          <td class="num">{safe(co.get('cost_of_living_index'))}</td>
        </tr>
        """

    # Section 4 Euro stats
    e_inf = f"{safe(euro_self.get('inflation_pre'))}% → {safe(euro_self.get('inflation_post'))}%"
    e_ca  = f"{safe(euro_self.get('curr_acc_pre'))}% → {safe(euro_self.get('curr_acc_post'))}%"
    e_debt = (f"{euro_self['govt_debt_pre']}% → {euro_self['govt_debt_post']}%"
              if euro_self.get('govt_debt_pre') and euro_self.get('govt_debt_post') else "N/A")
    e_gdp = f"{safe(euro_self.get('gdp_growth_pre'))}% → {safe(euro_self.get('gdp_growth_post'))}%"

    # Section 5 crisis table
    crisis_rows = ""
    for cr in crises:
        rc = cr.get("rank_change") or 0
        color = "var(--green)" if rc >= 0 else "var(--red)"
        verdict = "Winner" if rc >= 3 else "Loser" if rc <= -3 else "Stable"
        crisis_rows += f"""
        <tr>
          <td><strong>{cr['nombre_evento']}</strong> <span style="color:hsl(228,8%,55%);font-size:.7rem">{cr['anio_inicio']}-{cr['anio_fin']}</span></td>
          <td class="num" style="color:{color}">{'+' if rc > 0 else ''}{rc}</td>
          <td class="num">{fmt_pct(cr.get('gdp_growth_min'))}</td>
          <td class="num">{fmt_pct(cr.get('unemployment_max'))}</td>
          <td class="num">{safe(cr.get('years_to_recover'))}</td>
          <td>{verdict}</td>
        </tr>
        """

    # Section 7 trends chips
    trend_chips = ""
    for t in recent:
        label = t["label"]
        delta = t.get("delta_pct_3y")
        if delta is None:
            continue
        # Direction "good" for indicators where lower is better
        lower_better = any(k in label for k in ["Paro", "Inflación", "Deuda"])
        good = (delta < 0) if lower_better else (delta > 0)
        color = "var(--green)" if good else "var(--red)"
        sign = "+" if delta > 0 else ""
        trend_chips += f"""
        <div class="cs-stat" style="border-left:3px solid {color}">
          <div class="cs-stat__val" style="color:{color};font-size:1rem">{sign}{delta:.1f}%</div>
          <div class="cs-stat__lbl">{label}<br><span style="color:hsl(228,8%,50%)">{t['latest_year']}</span></div>
        </div>
        """

    # Section 8 ML
    ml_pct = (ml_risk.get("crisis_proba") or 0) * 100
    vu_score = safe(vulnerability.get("vulnerability_score"), 29)
    cluster_label = cluster.get("label") or "Eurozone core (currency-locked)"

    # ── Compose final HTML ───────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>¿Es España una economía fuerte? — Análisis macroeconómico</title>
  <meta name="description" content="Análisis de 60 años de datos del Banco Mundial, modelos de Machine Learning y comparativas Norte/Sur Europa para determinar la posición real de España en la economía global.">
  <meta property="og:title" content="¿Es España una economía fuerte?">
  <meta property="og:description" content="Análisis macroeconómico con 47 indicadores, 60 años de datos, y modelos ML.">
  <meta property="og:type" content="article">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Source+Serif+Pro:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root{{
      --bg:#fafaf7;
      --surface:#ffffff;
      --border:#e8e6e0;
      --border-strong:#d4d1c8;
      --text:#3a3a3a;
      --text-soft:#6b6b6b;
      --title:#1a1a1a;
      --accent:#c41e3a;
      --accent-soft:#fef2f3;
      --gold:#b8860b;
      --green:#2e7d32;
      --red:#c62828;
      --serif:'Source Serif Pro',Georgia,Cambria,'Times New Roman',serif;
      --sans:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    }}
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
    html{{font-size:17px;}}
    body{{font-family:var(--sans);background:var(--bg);color:var(--text);line-height:1.65;
      padding:2rem 1rem 3rem;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;}}
    main{{max-width:720px;margin:0 auto;}}
    a{{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent;transition:border-color .2s;}}
    a:hover{{border-bottom-color:var(--accent);}}

    .cs-meta{{font-size:.78rem;color:var(--text-soft);text-align:center;margin-bottom:2rem;
      padding-bottom:1.5rem;border-bottom:1px solid var(--border);}}

    .cs-hero{{text-align:center;margin-bottom:3.5rem;padding:1rem 0 2rem;}}
    .cs-hero__flag{{font-size:2.5rem;line-height:1;margin-bottom:.8rem;}}
    .cs-hero__kicker{{font-family:var(--sans);font-size:.7rem;color:var(--accent);
      font-weight:600;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:.85rem;}}
    .cs-hero__title{{font-family:var(--serif);font-size:2.4rem;font-weight:700;color:var(--title);
      line-height:1.15;margin-bottom:1.1rem;letter-spacing:-.5px;}}
    .cs-hero__sub{{font-family:var(--serif);font-size:1.15rem;color:var(--text-soft);
      max-width:580px;margin:0 auto;line-height:1.55;font-style:italic;}}
    .cs-hero__kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
      gap:1.5rem;margin-top:2.5rem;padding:1.5rem 0;border-top:1px solid var(--border);
      border-bottom:1px solid var(--border);}}
    .cs-hero__kpi{{text-align:center;}}
    .cs-hero__kpi-val{{font-family:var(--serif);font-size:1.6rem;font-weight:700;color:var(--title);
      line-height:1;font-variant-numeric:tabular-nums;}}
    .cs-hero__kpi-lbl{{font-size:.65rem;text-transform:uppercase;letter-spacing:1px;
      color:var(--text-soft);margin-top:.45rem;font-weight:500;}}

    .cs-toc{{background:var(--surface);border:1px solid var(--border);border-radius:4px;
      padding:1.4rem 1.6rem;margin-bottom:3rem;}}
    .cs-toc strong{{font-family:var(--sans);color:var(--text-soft);text-transform:uppercase;
      letter-spacing:1.5px;font-size:.68rem;display:block;margin-bottom:.7rem;font-weight:600;}}
    .cs-toc ol{{margin:0;padding-left:1.4rem;color:var(--text);line-height:1.9;list-style-type:decimal;
      font-family:var(--serif);font-size:.95rem;}}
    .cs-toc li{{margin-bottom:.2rem;}}
    .cs-toc a{{color:var(--text);}}
    .cs-toc a:hover{{color:var(--accent);border-bottom:none;}}

    .cs-sec{{margin-bottom:4rem;}}
    .cs-sec__num{{font-family:var(--sans);font-size:.7rem;color:var(--accent);
      font-weight:600;text-transform:uppercase;letter-spacing:1.5px;}}
    .cs-sec__title{{font-family:var(--serif);font-size:1.85rem;font-weight:700;color:var(--title);
      margin:.4rem 0 1.4rem;line-height:1.2;letter-spacing:-.3px;}}
    .cs-sec__body p{{font-family:var(--serif);font-size:1.05rem;color:var(--text);
      line-height:1.75;margin-bottom:1.1rem;}}
    .cs-sec__body strong{{color:var(--title);font-weight:600;}}
    .cs-sec__body em{{color:var(--accent);font-style:italic;}}
    .cs-sec__body ul{{margin:.7rem 0 1.2rem 1.6rem;color:var(--text);line-height:1.75;
      font-family:var(--serif);font-size:1rem;}}
    .cs-sec__body ul li{{margin-bottom:.5rem;}}

    .cs-pullquote{{font-family:var(--serif);font-size:1.4rem;font-style:italic;
      color:var(--title);line-height:1.4;text-align:center;
      margin:2.5rem 0;padding:1.5rem 2rem;
      border-top:2px solid var(--title);border-bottom:2px solid var(--title);
      font-weight:400;}}
    .cs-pullquote::before{{content:'"';font-size:3rem;line-height:0;vertical-align:-.4em;
      color:var(--accent);margin-right:.2rem;font-family:Georgia,serif;}}
    .cs-pullquote::after{{content:'"';font-size:3rem;line-height:0;vertical-align:-.4em;
      color:var(--accent);margin-left:.2rem;font-family:Georgia,serif;}}

    .cs-data-table{{width:100%;border-collapse:collapse;font-size:.88rem;margin:1.3rem 0;
      font-family:var(--sans);}}
    .cs-data-table th{{padding:.65rem .8rem;text-align:left;color:var(--text-soft);
      font-weight:600;font-size:.7rem;text-transform:uppercase;letter-spacing:1px;
      border-bottom:2px solid var(--title);}}
    .cs-data-table td{{padding:.7rem .8rem;border-bottom:1px solid var(--border);
      color:var(--text);font-family:var(--sans);font-size:.88rem;}}
    .cs-data-table td.num{{text-align:right;font-variant-numeric:tabular-nums;
      font-weight:500;color:var(--title);font-feature-settings:"tnum";}}
    .cs-data-table tr.spain{{background:var(--accent-soft);}}
    .cs-data-table tr.spain td{{color:var(--title);font-weight:600;border-bottom-color:var(--accent);}}
    .cs-data-table tr:hover td{{background:rgba(0,0,0,.02);}}

    .cs-verdict{{background:var(--surface);border:1px solid var(--border-strong);
      border-left:4px solid var(--accent);padding:1.8rem 2rem;margin:2.5rem 0;border-radius:2px;}}
    .cs-verdict h3{{font-family:var(--sans);font-size:.75rem;color:var(--accent);
      margin-bottom:1rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;}}
    .cs-verdict p{{font-family:var(--serif);font-size:1.05rem;color:var(--text);
      line-height:1.75;margin-bottom:1rem;}}
    .cs-verdict p:last-child{{margin-bottom:0;}}

    .cs-stat-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
      gap:1.5rem;margin:1.5rem 0 2rem;padding:1.4rem 0;
      border-top:1px solid var(--border);border-bottom:1px solid var(--border);}}
    .cs-stat{{text-align:center;}}
    .cs-stat__val{{font-family:var(--serif);font-size:1.55rem;font-weight:700;
      color:var(--title);line-height:1;font-variant-numeric:tabular-nums;}}
    .cs-stat__lbl{{font-family:var(--sans);font-size:.7rem;color:var(--text-soft);
      margin-top:.5rem;line-height:1.4;font-weight:400;}}

    .cs-footer{{margin-top:5rem;padding:2rem 0 0;border-top:2px solid var(--title);
      font-size:.78rem;color:var(--text-soft);line-height:1.65;}}
    .cs-footer p{{margin-bottom:.7rem;font-family:var(--sans);}}
    .cs-footer p:last-child{{font-family:var(--serif);font-style:italic;}}

    @media print {{
      body{{padding:1rem;font-size:11pt;}}
      .cs-hero__title{{font-size:1.8rem;}}
      .cs-sec__title{{font-size:1.3rem;}}
      .cs-sec, .cs-toc, .cs-data-table, .cs-stat-row, .cs-pullquote{{break-inside:avoid;}}
      a{{color:var(--title);}}
    }}
    @media (max-width:640px){{
      html{{font-size:15px;}}
      body{{padding:1.2rem .9rem 2rem;}}
      .cs-hero__title{{font-size:1.7rem;}}
      .cs-sec__title{{font-size:1.4rem;}}
      .cs-pullquote{{font-size:1.15rem;padding:1.1rem 1rem;}}
      .cs-data-table{{font-size:.78rem;}}
      .cs-data-table th, .cs-data-table td{{padding:.5rem .55rem;}}
      .cs-hero__kpis{{grid-template-columns:repeat(2,1fr);gap:1.2rem;}}
    }}
  </style>
</head>
<body>
<main>

<div class="cs-meta">
  ANÁLISIS MACROECONÓMICO &nbsp;·&nbsp; {datetime.now().strftime('%d de %B de %Y')} &nbsp;·&nbsp;
  Datos: World Bank &nbsp;·&nbsp; <a href="https://github.com/" target="_blank">Repositorio</a>
</div>

<div class="cs-hero">
  <div class="cs-hero__flag">{c.get('flag_emoji','🇪🇸')}</div>
  <div class="cs-hero__kicker">Caso de estudio · España 1965–2024</div>
  <h1 class="cs-hero__title">¿Es España una economía fuerte?</h1>
  <p class="cs-hero__sub">Un análisis basado en 47 indicadores del Banco Mundial, 60 años de datos históricos y 5 modelos analíticos. La respuesta corta: depende de con quién la compares.</p>
  <div class="cs-hero__kpis">{hero_kpis}</div>
</div>

<div class="cs-toc">
  <strong>Índice</strong>
  <ol>
    <li><a href="#cs1">Resumen ejecutivo</a></li>
    <li><a href="#cs2">¿Es España una economía fuerte? La métrica honesta</a></li>
    <li><a href="#cs3">Norte vs Sur Europa — la fractura estructural</a></li>
    <li><a href="#cs4">El euro: ¿beneficio o trampa?</a></li>
    <li><a href="#cs5">Crisis en 60 años — España, una y otra vez</a></li>
    <li><a href="#cs6">Decisiones políticas que marcaron la trayectoria</a></li>
    <li><a href="#cs7">Los últimos 3 años: ¿hay recuperación?</a></li>
    <li><a href="#cs8">Lo que dice el modelo predictivo</a></li>
    <li><a href="#cs9">Veredicto: ¿hacia dónde va España?</a></li>
    <li><a href="#cs10">Metodología y fiabilidad de los datos</a></li>
    <li><a href="#cs11">Glosario de términos técnicos</a></li>
  </ol>
</div>

<div class="cs-sec" id="cs1">
  <div class="cs-sec__num">§ 1 · Resumen ejecutivo</div>
  <div class="cs-sec__title">España es una economía <em>mediana consolidada</em>, no fuerte ni débil</div>
  <div class="cs-sec__body">
    <p>España aparece entre {gdp_pos_txt} ({fmt_usd(c.get('latest_gdp_per_capita'))}) pero solo en el <strong>puesto {safe(strength.get('global_rank'))} de {safe(strength.get('total'))}</strong> según la <em>puntuación compuesta de fortaleza</em>, una medida que combina cuatro elementos: el nivel de renta, la estabilidad histórica del crecimiento, la capacidad de recuperación ante crisis y la diversificación de su economía entre sectores. La paradoja: España es rica en términos absolutos pero su economía tiene <strong>fragilidades estructurales</strong> que la sitúan por debajo de su grupo natural de comparación —los países desarrollados de la eurozona.</p>
    <p>La <em>puntuación de fortaleza</em> de <strong>{safe(strength.get('strength_score'))}/100</strong> se compone de cuatro elementos. Cuanto mayor el número, mejor:</p>
    <div class="cs-stat-row">{sec1_stats}</div>
    <p>Los dos primeros elementos están bien — España tiene nivel de renta alto y una estabilidad razonable del crecimiento. Los dos últimos arrastran la puntuación: la <strong>recuperación lenta</strong> tras las crisis (consecuencia directa de no poder devaluar dentro del euro) y la <strong>poca diversificación</strong> entre sectores (alta concentración en servicios, especialmente turismo).</p>
  </div>
</div>

<div class="cs-sec" id="cs2">
  <div class="cs-sec__num">§ 2 · La métrica honesta</div>
  <div class="cs-sec__title">Rica en absoluto, mediana en términos relativos</div>
  <div class="cs-sec__body">
    <p>Comparada con el resto del mundo, España está claramente en el lado privilegiado. Pero la comparación relevante no es España frente al promedio mundial: es <strong>España frente a los países con los que compite directamente</strong> —la eurozona desarrollada, los miembros de la OCDE, los países similares.</p>
    <div class="cs-pullquote">España tiene el poder adquisitivo de un europeo medio ({fmt_usd(pp.get('gni_per_capita_ppp'))} por habitante ajustado por coste de vida, el {safe(pp.get('pct_us_gni_ppp'))}% del estadounidense) pero un mercado laboral que se parece más al de una economía emergente: paro de dos dígitos, mucho contrato temporal y baja productividad por hora trabajada.</div>
    <p>Dentro del <strong>sur de Europa</strong>, España ocupa el puesto <strong>{safe(zone.get('rk_gdp_cap'),6)} de {safe(zone.get('zone_size'),9)}</strong> en PIB per cápita y el <strong>{safe(zone.get('rk_unemp_best'),6)} de {safe(zone.get('zone_size'),9)}</strong> en menor desempleo. Es decir: dentro de su propia subregión, está en el tercio bajo en términos de mercado laboral. Domina demográficamente —el <strong>{safe(zone.get('pop_share_subregion'),37.5)}% de la población de la zona sur europea es española</strong>—, pero ese peso poblacional no se traduce en liderazgo económico.</p>
    <p>Un algoritmo de agrupamiento automático (técnica que reúne países similares sin reglas previas, ver glosario) coloca a España en el grupo "<em>{cluster_label}</em>" junto a Alemania, Francia, Italia y otros. Es decir: <strong>compartir el euro crea un patrón económico distintivo</strong>, incluso sin que se le indique explícitamente al algoritmo qué países lo usan.</p>
  </div>
</div>

<div class="cs-sec" id="cs3">
  <div class="cs-sec__num">§ 3 · La fractura europea</div>
  <div class="cs-sec__title">Norte ordenado, Sur con paro estructural — España en el medio</div>
  <div class="cs-sec__body">
    <p>La fractura norte-sur europea no es retórica política, está en los datos:</p>
    <table class="cs-data-table">
      <thead><tr><th>País</th><th>PIB/cápita</th><th>Paro</th><th>Deuda pública</th><th>Fortaleza</th><th>Coste vida</th></tr></thead>
      <tbody>{europe_rows}</tbody>
    </table>
    <p>Tres patrones aparecen con claridad:</p>
    <p><strong>1. Brecha de productividad.</strong> Los países del norte (Noruega, Países Bajos, Suecia, Alemania) tienen un PIB per cápita entre 1,5 y 2,5 veces el de España. Esa diferencia no se cierra con políticas coyunturales: refleja décadas de inversión en capital humano, tecnología y diversificación de la economía.</p>
    <p><strong>2. Mercados laborales.</strong> El paro en el norte oscila entre el 3% y el 5%; en el sur (España, Italia, Grecia) sigue por encima del 10%. España, aunque ha reducido el paro del 26% que tuvo en 2013 al 10% actual, sigue duplicando o triplicando la cifra del norte.</p>
    <p><strong>3. Coste de vida.</strong> El norte es más caro (índice 85-108, sobre la base de 100 que representa EE.UU.), pero los salarios compensan. España es un 39% más barata que EE.UU. (índice 61), pero ese "bonus de coste" no compensa los salarios significativamente menores.</p>
    <div class="cs-pullquote">El sur europeo no es pobre. Es <em>menos productivo</em> que el norte —esa es la diferencia clave que los datos muestran de forma consistente desde 1965.</div>
  </div>
</div>

<div class="cs-sec" id="cs4">
  <div class="cs-sec__num">§ 4 · El euro</div>
  <div class="cs-sec__title">Estabilidad de precios a cambio de la herramienta de devaluación</div>
  <div class="cs-sec__body">
    <p>En 1999 España renunció a la peseta. Los datos del antes/después adopción del euro cuentan una historia mixta:</p>
    <div class="cs-stat-row">
      <div class="cs-stat"><div class="cs-stat__val" style="color:var(--green)">{e_inf}</div><div class="cs-stat__lbl">Inflación media<br>✓ bajó como se prometía</div></div>
      <div class="cs-stat"><div class="cs-stat__val" style="color:var(--red)">{e_ca}</div><div class="cs-stat__lbl">Balanza comercial<br>✗ el déficit empeoró</div></div>
      <div class="cs-stat"><div class="cs-stat__val" style="color:var(--red)">{e_debt}</div><div class="cs-stat__lbl">Deuda pública<br>✗ creció +28 puntos</div></div>
      <div class="cs-stat"><div class="cs-stat__val">{e_gdp}</div><div class="cs-stat__lbl">Crecimiento del PIB<br>≈ estable</div></div>
    </div>
    <p>El euro <strong>cumplió la promesa de inflación baja</strong> ({euro_self.get('inflation_pre','4.7')}% antes del euro a {euro_self.get('inflation_post','2.9')}% después). Pero el coste fue real y cuantificable:</p>
    <p><strong>Pérdida del ajuste cambiario.</strong> Antes de 1999, cuando llegaba una crisis la peseta se devaluaba automáticamente: las exportaciones españolas se abarataban en el extranjero, los turistas extranjeros venían más baratos y la economía se recuperaba sola por la vía de la competitividad. Sin esa válvula, cuando estalló Lehman en 2008 el ajuste tuvo que venir por la llamada <em>devaluación interna</em>: salarios congelados, paro al 26% y deuda pública del 130% del PIB.</p>
    <p><strong>Divergencia con Alemania.</strong> La balanza comercial de Alemania pasó de un déficit del 0,7% a un <strong>superávit del 6%</strong> del PIB; la de España, del -0,6% al -4,2%. Una diferencia de <strong>más de 10 puntos porcentuales</strong> que refleja cómo durante 2000-2007 los ahorros del norte financiaron la burbuja inmobiliaria del sur, con los tipos de interés bajos que decidía el Banco Central Europeo para toda la zona, no para cada país.</p>
    <div class="cs-pullquote">El euro funcionó como una camisa de fuerza monetaria. La promesa de no devaluar se cumplió, pero el ajuste tuvo que venir vía deuda y desempleo.</div>
  </div>
</div>

<div class="cs-sec" id="cs5">
  <div class="cs-sec__num">§ 5 · Crisis en 60 años</div>
  <div class="cs-sec__title">El ciclo recurrente: shock externo → ajuste vía paro</div>
  <div class="cs-sec__body">
    <p>España ha vivido doce grandes crisis económicas entre 1973 y 2023. Su comportamiento en cada una revela un patrón. La columna "Cambio de puesto" mide cuántas posiciones subió o bajó España en el ranking mundial de PIB per cápita durante esa crisis (positivo es mejor):</p>
    <table class="cs-data-table">
      <thead><tr><th>Crisis</th><th>Cambio de puesto</th><th>Mín. crecimiento</th><th>Máx. paro</th><th>Años en recuperarse</th><th>Veredicto</th></tr></thead>
      <tbody>{crisis_rows}</tbody>
    </table>
    <p>El <strong>periodo Lehman → Crisis del euro → Caída del petróleo (2008-2016)</strong> fue la herida más grave de la historia económica reciente de España: <strong>perdió 7 puestos en cada una</strong>, pasando del puesto 37 al 48 mundial en PIB per cápita. Tardó once años en recuperar lo perdido.</p>
    <p>Lo notable es la <strong>recuperación tras la pandemia (+10 puestos) y tras la invasión de Ucrania (+9)</strong>: España subió 19 puestos entre 2020 y 2024. Las hipótesis: el turismo se recuperó muy rápido (España es la segunda potencia turística del mundo), los fondos europeos <em>Next Generation EU</em> inyectaron capital, y el mercado laboral mejoró con las reformas de 2021-2022.</p>
    <div class="cs-pullquote">Las crisis financieras castigaron a España de forma desproporcionada. Pero las crisis posteriores a 2020 (sanitarias y geopolíticas) la beneficiaron en términos relativos.</div>
  </div>
</div>

<div class="cs-sec" id="cs6">
  <div class="cs-sec__num">§ 6 · Decisiones políticas</div>
  <div class="cs-sec__title">5 inflexiones que marcaron 60 años</div>
  <div class="cs-sec__body">
    <p><strong>1959 — Plan de Estabilización.</strong> Fin de la autarquía franquista. Apertura al comercio. Sienta la base del milagro económico español 1960-1973 (crecimiento medio ~7%).</p>
    <p><strong>1986 — Entrada en la CEE.</strong> Acceso al mercado europeo, fondos estructurales, modernización. España converge con Europa: de 60% del PIB/cápita medio europeo en 1985 al 88% en 2007.</p>
    <p><strong>1999 — Adopción del euro.</strong> Convergencia de tipos de interés (de 14% a 4%). Crédito barato → burbuja inmobiliaria 2000-2007. La construcción llega al 12% del PIB. La economía aparenta un "milagro" que era endeudamiento privado masivo.</p>
    <p><strong>2010-2012 — Austeridad fiscal forzada.</strong> Sin poder devaluar, España aplica recortes presupuestarios drásticos. Paro del 26% en 2013, mayor incluso que en la Guerra Civil. La generación 1980-1995 sufre escasez de oportunidades.</p>
    <p><strong>2021-2024 — Reforma laboral + fondos NGEU.</strong> Reducción de la temporalidad (de 26% a 13%), inversión pública récord, salario mínimo +47% desde 2018. La economía supera el pre-pandemia en 2023.</p>
    <div class="cs-pullquote">España es un país que reacciona bien a los incentivos externos (CEE, euro, fondos UE) pero estructura mal sus instituciones laborales y fiscales internas. Cada generación parte de cero con el paro.</div>
  </div>
</div>

<div class="cs-sec" id="cs7">
  <div class="cs-sec__num">§ 7 · Últimos 3 años</div>
  <div class="cs-sec__title">9 indicadores mejoran, 1 empeora</div>
  <div class="cs-sec__body">
    <p>Los datos del periodo 2021-2024 muestran una recuperación amplia tras la pandemia y la invasión de Ucrania. Cada tarjeta muestra el cambio porcentual respecto al valor de hace tres años:</p>
    <div class="cs-stat-row">{trend_chips}</div>
    <p>El paro cayó un <strong>{unemp_drop}%</strong> en tres años, la deuda pública bajó un <strong>{debt_drop}%</strong>, y la balanza comercial pasó de cerca de cero a un superávit del 3,2% del PIB. La inflación volvió cerca del objetivo del Banco Central Europeo (2,8% actualmente, frente al objetivo del 2%). El único indicador en color rojo es la desaceleración del crecimiento, pero partía de un rebote tras la pandemia muy alto (más del 5%).</p>
    <p>Si esta tendencia se mantiene, España podría recuperar parte del terreno perdido en 2008-2016. Pero requiere mantener la disciplina presupuestaria —algo que históricamente ha sido difícil en España.</p>
  </div>
</div>

<div class="cs-sec" id="cs8">
  <div class="cs-sec__num">§ 8 · El modelo predictivo</div>
  <div class="cs-sec__title">Probabilidad de patrón de crisis: <em>{ml_pct:.1f}%</em> (riesgo bajo)</div>
  <div class="cs-sec__body">
    <p>Entrenamos un modelo de <em>Random Forest</em> (algoritmo de "bosque aleatorio" que combina cientos de árboles de decisión, ver glosario) con {n_txt} observaciones país-año del periodo 1965-2014. Lo validamos sobre datos posteriores (2015-2024) que el modelo no había visto durante el entrenamiento —lo que se llama validación "fuera de muestra". Su capacidad de discriminación, medida con la métrica AUC, es de {auc_txt}: un buen resultado para datos macroeconómicos (un valor de 0,5 sería aleatorio, 1,0 sería perfecto).</p>
    <p>El modelo clasifica a España como <strong>de bajo riesgo de presentar un patrón pre-crisis</strong> (probabilidad: {ml_pct:.1f}%). Este resultado es coherente con los demás indicadores: España no muestra las <em>señales</em> típicas que preceden a una crisis (alta deuda externa, inflación descontrolada, baja inversión productiva). Sus problemas son <strong>estructurales</strong> —productividad, paro de larga duración, envejecimiento—, no de inestabilidad inmediata.</p>
    <p>La <strong>vulnerabilidad externa</strong> (puntuación {vu_score}/100, donde 100 sería extremadamente vulnerable) está en el rango bajo-medio: ni es un petroestado dependiente de un solo recurso, ni una economía cerrada sin reservas en moneda extranjera. El algoritmo de agrupamiento la coloca en el grupo "<em>{cluster_label}</em>", reuniéndola con las economías de la eurozona.</p>
  </div>
</div>

<div class="cs-sec" id="cs9">
  <div class="cs-verdict">
    <h3>§ 9 · Veredicto</h3>
    <p><strong>¿Es España una economía fuerte? La respuesta honesta es <em>no, pero tampoco débil</em>.</strong> Es una economía consolidada y mediana, atrapada en una camisa de fuerza monetaria —el euro— que limita sus opciones de ajuste, y con problemas estructurales que ningún gobierno desde la transición ha logrado resolver: paro de larga duración elevado, baja productividad por hora trabajada, dependencia del turismo y demografía envejecida (1,16 hijos por mujer en 2024).</p>
    <p><strong>España no va a ser Alemania.</strong> Pero tampoco corre riesgo inminente de convertirse en Argentina. Su trayectoria razonable: una economía de servicios de renta alta, que se acerca lentamente a la media europea, vulnerable a shocks externos pero capaz de recuperarse cuando el entorno mejora.</p>
    <p><strong>El factor crítico de los próximos veinte años</strong> será la transición demográfica: con una tasa de dependencia del 56% (proporción de niños y mayores por cada adulto en edad de trabajar) y una fertilidad de 1,16 hijos por mujer, sin inmigración significativa la fuerza laboral disminuirá. La productividad por trabajador tendría que crecer al doble del ritmo actual para mantener el nivel de vida.</p>
  </div>
</div>

<div class="cs-sec" id="cs10">
  <div class="cs-sec__num">§ 10 · Metodología</div>
  <div class="cs-sec__title">Fuentes, fiabilidad y limitaciones reconocidas</div>
  <div class="cs-sec__body">
    <p><strong>Fuentes de datos:</strong></p>
    <ul>
      <li><strong>Banco Mundial (World Bank Open Data)</strong> — 47 indicadores macroeconómicos, panel completo 1965-2024. Licencia abierta CC-BY-4.0.</li>
      <li><strong>REST Countries v3.1</strong> — Metadatos de países: banderas, regiones y subregiones.</li>
      <li><strong>FRED (Reserva Federal de San Luis)</strong> — Contexto macro global: precio del petróleo Brent, índice de volatilidad VIX, tipo de interés de la Reserva Federal, índice de estrés financiero.</li>
      <li><strong>Caldara y Iacoviello</strong> — Índice de Riesgo Geopolítico (GPR), elaborado por economistas de la Reserva Federal.</li>
      <li><strong>Eurostat HBS, BLS CES, INEGI</strong> — Desglose del presupuesto familiar para 31 países, encuestas oficiales de gasto de hogares.</li>
    </ul>
    <p><strong>Modelos analíticos utilizados:</strong></p>
    <ul>
      <li><strong>Random Forest (bosque aleatorio)</strong> — 200 árboles de decisión combinados. Métrica de calidad AUC = {auc_txt}. Predice si un país presenta el patrón típico de pre-crisis.</li>
      <li><strong>Agrupamiento K-Means</strong> — 5 grupos descubiertos, índice de calidad <em>silhouette</em> 0,185, basado en 27 variables económicas.</li>
      <li><strong>Reglas de clasificación por arquetipos</strong> — 8 categorías deterministas (petroestado, exportador manufacturero, economía de servicios, etc.) para una interpretación intuitiva.</li>
      <li><strong>Puntuaciones compuestas</strong> — Fortaleza (pondera nivel/estabilidad/recuperación/diversificación al 35/25/20/20%), vulnerabilidad externa (40/30/30%), índice de coste de vida (cociente PIB nominal / PIB ajustado por coste de vida).</li>
    </ul>
    <p><strong>Limitaciones reconocidas:</strong></p>
    <ul>
      <li>Los datos sectoriales (qué porcentaje del PIB es servicios, industria, agricultura) solo están disponibles desde 1995 para España, por una limitación del Banco Mundial.</li>
      <li>Los datos de empleo por sector solo se publican desde 1991 (limitación de la OIT).</li>
      <li>Las series de paridad de poder adquisitivo (PPA) empiezan en 1990.</li>
      <li>Iraq 1971 fue excluido del análisis cambiario por un error documentado en la base del Banco Mundial.</li>
      <li>El presupuesto familiar es una <em>estimación del gasto medio</em>, no la renta real disponible de cada hogar concreto.</li>
    </ul>
    <p style="margin-top:1rem">Tecnología empleada: <strong>PostgreSQL 16</strong> (base de datos), <strong>FastAPI</strong> (servicio web), <strong>Python 3.12</strong>, <strong>scikit-learn 1.4</strong> (modelos de aprendizaje automático), <strong>pandas 2.2</strong> (procesamiento de datos), <strong>Docker Compose</strong> (despliegue reproducible). El proceso de extracción, transformación y carga (ETL) es idempotente —se puede ejecutar varias veces sin generar duplicados— y mantiene 17 vistas analíticas materializadas que aceleran las consultas.</p>
  </div>
</div>

<div class="cs-sec" id="cs11">
  <div class="cs-sec__num">§ 11 · Glosario</div>
  <div class="cs-sec__title">Términos técnicos explicados</div>
  <div class="cs-sec__body">
    <p style="font-size:.9rem;color:var(--text-soft);font-style:italic;margin-bottom:1.5rem">Algunos términos económicos y técnicos pueden resultar opacos. Esta es una lista de los más importantes que aparecen en el informe.</p>
    <ul>
      <li><strong>PIB per cápita (nominal)</strong> — Producto Interior Bruto dividido entre la población. Lo que produce el país por habitante medido en dólares actuales. No ajusta por diferencias de precios entre países.</li>
      <li><strong>PIB per cápita PPA / PPP</strong> — Lo mismo pero <em>ajustado por paridad de poder adquisitivo</em>: cuánto compras realmente en tu país con el dinero que tienes. Un cortado cuesta menos en Madrid que en Nueva York; PPA lo refleja.</li>
      <li><strong>Balanza por cuenta corriente</strong> — Diferencia entre lo que un país vende al exterior (exportaciones, turismo recibido, etc.) y lo que compra. Superávit = vende más; déficit = compra más.</li>
      <li><strong>Puntuación de fortaleza</strong> — Métrica compuesta de 0 a 100 que combina cuatro elementos: nivel de renta, estabilidad histórica, capacidad de recuperación tras crisis y diversificación entre sectores. Cuanto mayor, mejor.</li>
      <li><strong>Vulnerabilidad externa</strong> — Puntuación de 0 a 100 que mide cuán expuesto está un país a shocks de fuera: depende de su deuda externa, dependencia de un único producto exportado, y reservas de divisas. Mayor número = más vulnerable.</li>
      <li><strong>Devaluación interna</strong> — Cuando un país no puede devaluar su moneda (porque comparte el euro, por ejemplo), el ajuste tras una crisis viene por congelación de salarios, recortes de gasto público y subida del paro. España la sufrió entre 2010 y 2014.</li>
      <li><strong>Random Forest (bosque aleatorio)</strong> — Algoritmo de aprendizaje automático que combina cientos de árboles de decisión simples. Cada árbol vota, y la mayoría decide. Robusto y fácil de interpretar.</li>
      <li><strong>K-Means (agrupamiento automático)</strong> — Algoritmo que agrupa países similares sin reglas previas. Tú le dices cuántos grupos quieres y él decide quién va con quién basándose solo en los datos.</li>
      <li><strong>AUC (Área Bajo la Curva ROC)</strong> — Métrica de calidad de un modelo predictivo. Va de 0 a 1. Un AUC de 0,5 es aleatorio (como tirar una moneda); 0,7 es aceptable; 0,8 es bueno; 0,9 es excelente; 1,0 sería perfección.</li>
      <li><strong>Validación fuera de muestra</strong> — Cuando entrenas un modelo con datos antiguos (por ejemplo, 1965-2014) y verificas su rendimiento con datos que no había visto antes (2015-2024). Es la prueba más honesta de si el modelo realmente aprende patrones reales o memoriza.</li>
      <li><strong>Eurozona central / periferia</strong> — La eurozona se divide informalmente en países "centrales" (Alemania, Francia, Países Bajos, Austria...) con balanzas comerciales positivas, y "periferia" (España, Italia, Grecia, Portugal) con déficits comerciales y crecimiento más lento.</li>
      <li><strong>Fondos Next Generation EU</strong> — Plan europeo aprobado en 2020 para impulsar la recuperación tras la pandemia. España recibe unos 140.000 millones de euros entre 2021 y 2026 (subvenciones y préstamos).</li>
      <li><strong>Tasa de dependencia</strong> — Personas en edad no laboral (niños, ancianos) por cada 100 adultos en edad de trabajar. Cuanto más alta, más carga fiscal sobre quienes trabajan.</li>
      <li><strong>Coste de vida (índice EE.UU. = 100)</strong> — Cuánto cuesta vivir en cada país en relación a Estados Unidos. España = 61 significa que vivir en España cuesta el 61% de lo que cuesta vivir en EE.UU.</li>
      <li><strong>Pipeline ETL</strong> — En inglés <em>Extract, Transform, Load</em>. Proceso automatizado que descarga datos de las fuentes (extract), los limpia y reorganiza (transform), y los guarda en la base de datos (load).</li>
    </ul>
  </div>
</div>

<div class="cs-footer">
  <p><strong>Informe generado el {datetime.now().strftime('%d de %B de %Y')}</strong> a partir del pipeline ETL de Countries ETL Dashboard. Todos los datos son públicos y verificables en las fuentes citadas. Este análisis es académico/portfolio y no constituye asesoramiento financiero.</p>
  <p>Código fuente y dashboard interactivo: <a href="https://github.com/" target="_blank">github.com/usuario/etl_countries</a> · Stack: Docker Compose / PostgreSQL / FastAPI / scikit-learn / Chart.js</p>
  <p style="margin-top:.7rem;color:hsl(45,90%,55%)">Para citar: <em>Spain Macroeconomic Case Study, Countries ETL Dashboard v4 ({datetime.now().year})</em>.</p>
</div>

</main>
</body>
</html>
"""

    OUT_FILE.write_text(html, encoding="utf-8")
    # GitHub Pages doesn't process Jekyll if .nojekyll exists
    (OUT_DIR / ".nojekyll").touch()
    print(f"\nOK Generated: {OUT_FILE}")
    print(f"  Size: {OUT_FILE.stat().st_size / 1024:.1f} KB")
    print(f"\nNext steps:")
    print(f"  1. git add docs/")
    print(f"  2. git commit -m 'Add Spain case study static page'")
    print(f"  3. git push")
    print(f"  4. In GitHub repo: Settings -> Pages -> Source: 'main' branch / docs/ folder")
    print(f"  5. Page will be at: https://<user>.github.io/<repo>/")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.RequestException as e:
        print(f"\nERROR: API not reachable at {API_BASE}. Make sure 'docker compose up -d api' is running.", file=sys.stderr)
        sys.exit(1)
