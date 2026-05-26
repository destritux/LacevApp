import pandas as pd
from pathlib import Path
import glob
from PIL import Image

def generate_html_report(base_path, experiment_description, gui_log_callback):
    """Gera relatório HTML completo com estatísticas clássicas OLS, resumos CSV na pasta 'results' e gráficos."""
    base_path = Path(base_path)
    features_path = base_path / "features"
    graphics_path = base_path / "graphics"
    results_path = base_path / "results"
    
    # Cria pasta de resultados se não existir
    results_path.mkdir(exist_ok=True)

    gui_log_callback("Carregando dados processados...")
    class_files = glob.glob(str(features_path / "df_*.csv"))
    if not class_files:
        gui_log_callback("Erro: Nenhum ficheiro df_*.csv encontrado.")
        return

    all_data = {}
    master_df_list = []
    
    for f in class_files:
        class_name = Path(f).stem.replace('df_', '')
        df = pd.read_csv(f)

        drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")
                     or c.lower() in ["classe", "nameclass", "date"]]
        df = df.drop(columns=drop_cols, errors="ignore")

        all_data[class_name] = df
        
        # Preparação do Master DF para regressão
        df_reg = df.copy()
        df_reg['TreatmentClass'] = class_name
        master_df_list.append(df_reg)

    gui_log_callback("Calculando estatísticas de resumo...")
    numeric_cols = all_data[list(all_data.keys())[0]].select_dtypes(include='number').columns.drop('minute', errors='ignore')
    summary_stats = {'Classe': []}
    for col in numeric_cols:
        summary_stats[f'{col}'] = []

    for class_name, df in all_data.items():
        summary_stats['Classe'].append(class_name)
        for col in numeric_cols:
            summary_stats[f'{col}'].append(df[col].mean())

    summary_df = pd.DataFrame(summary_stats).set_index('Classe')
    comparison_df = (summary_df.pct_change() * 100).iloc[[-1]]
    comparison_df = comparison_df.rename(index={comparison_df.index[0]: f'% Mudança ({comparison_df.index[0]} vs {summary_df.index[0]})'})

    summary_df_transposed = summary_df.T
    comparison_df_transposed = comparison_df.T

    # Salva resumos estatísticos em formato CSV na pasta 'results'
    summary_df_transposed.to_csv(results_path / "summary_stats.csv")
    comparison_df_transposed.to_csv(results_path / "comparison_stats.csv")

    # --- MODELO LINEAR (OLS) PAREADO COM STATSMODELS ---
    ols_html = ""
    highlight_html = ""
    try:
        gui_log_callback("Tentando processar estatísticas de Regressão Linear (OLS) pareadas...")
        import statsmodels.formula.api as smf
        import numpy as np
        import itertools
        
        master_df = pd.concat(master_df_list, ignore_index=True)
        z_cols = [c for c in master_df.columns if c.endswith('_Z')]
        classes = sorted(all_data.keys())
        
        if len(classes) >= 2 and z_cols:
            pairs = list(itertools.combinations(classes, 2))
            pairwise_results = []
            pair_summaries = []
            
            for classA, classB in pairs:
                # Filtrar o master_df para conter apenas observações desses dois grupos
                pair_df = master_df[master_df['TreatmentClass'].isin([classA, classB])].copy()
                pair_df['TreatmentClass'] = pd.Categorical(pair_df['TreatmentClass'], categories=[classA, classB], ordered=True)
                
                accepted_count = 0
                rejected_count = 0
                
                for z_col in z_cols:
                    safe_col = z_col.replace('-', '_')
                    temp_df = pair_df.rename(columns={z_col: safe_col})
                    temp_df = temp_df.dropna(subset=[safe_col, 'TreatmentClass'])
                    
                    if len(temp_df['TreatmentClass'].unique()) < 2:
                        continue
                        
                    try:
                        formula = f"{safe_col} ~ C(TreatmentClass)"
                        md = smf.ols(formula, temp_df)
                        mdf = md.fit()
                        
                        term = f"C(TreatmentClass)[T.{classB}]"
                        if term in mdf.pvalues:
                            p_value = mdf.pvalues[term]
                            coef = mdf.params[term]
                        else:
                            treatment_terms = [k for k in mdf.pvalues.index if k != 'Intercept']
                            if treatment_terms:
                                term = treatment_terms[0]
                                p_value = mdf.pvalues[term]
                                coef = mdf.params[term]
                            else:
                                p_value = np.nan
                                coef = np.nan
                                
                        is_sig = not pd.isna(p_value) and p_value < 0.05
                        sig = "Sim" if is_sig else "Não"
                        if is_sig:
                            accepted_count += 1
                        else:
                            rejected_count += 1
                            
                        pairwise_results.append({
                            'Classe_A': classA,
                            'Classe_B': classB,
                            'Métrica Z-Score': z_col,
                            'Efeito (Coef)': coef,
                            'P-Value': p_value,
                            'Significativo?': sig
                        })
                    except Exception as e:
                        gui_log_callback(f"Aviso no cálculo OLS de {z_col} para {classA} vs {classB}: {e}")
                
                pair_summaries.append({
                    'Par': f"{classA} vs {classB}",
                    'Classe_A': classA,
                    'Classe_B': classB,
                    'Características Significativas (Aceito)': accepted_count,
                    'Características Não Significativas (Rejeitado)': rejected_count
                })
            
            # Ordena os resumos pelo número de características significativas (Aceito) decrescente
            pair_summaries_df = pd.DataFrame(pair_summaries)
            pair_summaries_df = pair_summaries_df.sort_values(by='Características Significativas (Aceito)', ascending=False)
            
            # Identifica o par com maior diferença
            if not pair_summaries_df.empty:
                most_diff_row = pair_summaries_df.iloc[0]
                most_diff_pair = most_diff_row['Par']
                most_diff_count = most_diff_row['Características Significativas (Aceito)']
                highlight_html = f"""
                <div class="highlight-box">
                    <h3>📢 Destaque de Divergência Biológica</h3>
                    <p>O par de classes com <b>maior diferença geral</b> é <b>{most_diff_pair}</b>, apresentando <b>{most_diff_count}</b> características significativas (p < 0.05).</p>
                </div>
                """
            
            # Salvar resultados consolidados
            pairwise_df = pd.DataFrame(pairwise_results)
            pairwise_df.to_csv(results_path / "pairwise_comparisons.csv", index=False)
            pair_summaries_df.to_csv(results_path / "pairwise_summary.csv", index=False)
            
            # Construir HTML das tabelas OLS
            ols_html = "<h3>Resumo de Comparação Pareada (Ranking de Divergência)</h3>"
            ols_html += "<table><thead><tr><th>Par de Classes</th><th>Features Aceitas (Significativo, p<0.05)</th><th>Features Rejeitadas (Não Significativo)</th></tr></thead><tbody>"
            for _, row in pair_summaries_df.iterrows():
                ols_html += f"<tr><td><b>{row['Par']}</b></td><td><span class='badge badge-success'>{row['Características Significativas (Aceito)']}</span></td><td><span class='badge badge-neutral'>{row['Características Não Significativas (Rejeitado)']}</span></td></tr>"
            ols_html += "</tbody></table>"
            
            ols_html += "<h3>Tabelas de Regressão OLS Detalhadas por Par</h3>"
            for _, row_summary in pair_summaries_df.iterrows():
                p_name = row_summary['Par']
                cA = row_summary['Classe_A']
                cB = row_summary['Classe_B']
                
                # Filtrar resultados desse par
                res_filtered = [r for r in pairwise_results if r['Classe_A'] == cA and r['Classe_B'] == cB]
                if not res_filtered:
                    continue
                
                # Gerar tabela HTML
                table_rows = ""
                for r in res_filtered:
                    p_val = r['P-Value']
                    p_str = "" if pd.isna(p_val) else (f"{p_val:.4e}" if p_val < 0.0001 else f"{p_val:.4f}")
                    coef_str = "" if pd.isna(r['Efeito (Coef)']) else f"{r['Efeito (Coef)']:.4f}"
                    badge = f"<span class='badge badge-success'>Sim</span>" if r['Significativo?'] == "Sim" else f"<span class='badge badge-neutral'>Não</span>"
                    table_rows += f"<tr><td><b>{r['Métrica Z-Score']}</b></td><td>{coef_str}</td><td>{p_str}</td><td>{badge}</td></tr>"
                
                pair_table = f"""
                <table>
                    <thead>
                        <tr>
                            <th>Métrica Z-Score</th>
                            <th>Efeito (Coeficiente)</th>
                            <th>P-Value</th>
                            <th>Significativo? (p < 0.05)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
                """
                
                ols_html += f"""
                <details>
                    <summary>{p_name} — ({row_summary['Características Significativas (Aceito)']} features significativas)</summary>
                    <div class="details-content">
                        <p>Modelo OLS ajustado: <code>Feature_Z ~ C(TreatmentClass)</code> restringido aos dados de <b>{cA}</b> e <b>{cB}</b> (referência: <b>{cA}</b>).</p>
                        {pair_table}
                    </div>
                </details>
                """
        else:
            ols_html = "<p><i>Dados insuficientes ou colunas Z-Score não encontradas para regressão pareada.</i></p>"
            
    except ImportError:
        ols_html = "<p><i>Biblioteca 'statsmodels' não instalada. Análise OLS ignorada.</i></p>"
    except Exception as e:
        ols_html = f"<p><i>Erro no cálculo OLS: {e}</i></p>"

    # --- LÓGICA DE INTERPRETAÇÃO SIMPLES (SEM LLM) ---
    interpretations = []
    for col in summary_df.columns:
        try:
            best_class = summary_df[col].idxmax()
            max_val = summary_df[col].max()
            interpretations.append(f"O valor médio mais alto para a característica <b>{col}</b> foi <b>{max_val:.4f}</b> para a classe <b>{best_class}</b>.")
        except Exception:
            pass

    interpretations_html = ""
    if interpretations:
        list_items = "".join([f"<li>{item}</li>" for item in interpretations])
        interpretations_html = f"""
        <div class="section interpretations">
            <h2>Interpretação das Características (Resumo Estatístico)</h2>
            <ul>
                {list_items}
            </ul>
        </div>
        """

    gui_log_callback("Buscando gráficos...")
    graph_files = glob.glob(str(graphics_path / "*_light.png"))
    if not graph_files:
        gui_log_callback("Aviso: gráficos no tema claro não encontrados, tentando tema escuro...")
        graph_files = glob.glob(str(graphics_path / "*_dark.png"))

    if not graph_files:
        gui_log_callback("Aviso: Nenhum gráfico encontrado.")
        graph_files = []

    graph_files.sort()

    gui_log_callback("Construindo HTML final...")
    graphs_html = ""
    if graph_files:
        graphs_html = '<div class="grid-2">' + ''.join([
            f'<div class="img-container">'
            f'<div class="img-title">{Path(g).name.replace("_light.png", "").replace("_dark.png", "").replace("TDAF_", "")}</div>'
            f'<img src="../graphics/{Path(g).name}" alt="Gráfico de {Path(g).stem}">'
            f'</div>'
            for g in graph_files
        ]) + '</div>'
    else:
        graphs_html = "<p><i>Nenhum gráfico disponível.</i></p>"

    html = f"""
    <html>
    <head>
        <title>Relatório de Análise Eletrofisiológica</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

            :root {{
                --primary: #059669;
                --primary-dark: #047857;
                --primary-light: #ecfdf5;
                --secondary: #0d9488;
                --accent: #d97706;
                --bg-main: #f8fafc;
                --bg-card: #ffffff;
                --text-primary: #0f172a;
                --text-secondary: #475569;
                --border: #e2e8f0;
                --radius-md: 8px;
                --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            }}

            body {{
                font-family: 'Inter', sans-serif;
                background-color: var(--bg-main);
                color: var(--text-primary);
                line-height: 1.5;
                margin: 0;
                padding: 20px;
            }}

            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: var(--bg-card);
                padding: 40px;
                border-radius: 12px;
                box-shadow: var(--shadow);
                border: 1px solid var(--border);
            }}

            h1, h2, h3, h4 {{
                font-family: 'Outfit', sans-serif;
                color: #064e3b;
            }}

            h1 {{
                font-size: 2.2rem;
                font-weight: 800;
                margin-top: 0;
                border-bottom: 3px solid var(--primary);
                padding-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 12px;
            }}

            h2 {{
                font-size: 1.5rem;
                border-bottom: 2px solid var(--border);
                padding-bottom: 8px;
                margin-top: 30px;
            }}

            h3 {{
                font-size: 1.15rem;
                margin-top: 20px;
            }}

            /* Custom Tables */
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 30px;
                font-size: 0.9rem;
            }}

            th, td {{
                padding: 12px 14px;
                text-align: left;
                border-bottom: 1px solid var(--border);
            }}

            th {{
                background-color: #f1f5f9;
                color: #1e293b;
                font-weight: 600;
            }}

            tr:hover td {{
                background-color: #f8fafc;
            }}

            /* Experiment description style */
            .section.experiment-desc {{
                background-color: #f0fdf4;
                padding: 20px;
                border-left: 5px solid var(--primary);
                border-radius: var(--radius-md);
                margin-bottom: 30px;
            }}

            /* Highlight box for key results */
            .highlight-box {{
                background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
                border-left: 5px solid var(--primary);
                border-radius: var(--radius-md);
                padding: 20px;
                margin-bottom: 30px;
                color: #064e3b;
            }}

            .highlight-box p {{
                margin: 6px 0;
                font-size: 1.05rem;
            }}

            /* Badges */
            .badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}

            .badge-success {{
                background-color: #d1fae5;
                color: #065f46;
            }}

            .badge-neutral {{
                background-color: #f1f5f9;
                color: #475569;
            }}

            /* Details / Accordion */
            details {{
                border: 1px solid var(--border);
                border-radius: var(--radius-md);
                margin-bottom: 15px;
                background-color: #ffffff;
                overflow: hidden;
            }}

            summary {{
                padding: 14px 20px;
                font-weight: 600;
                font-family: 'Outfit', sans-serif;
                cursor: pointer;
                background-color: #f8fafc;
                user-select: none;
                transition: background-color 0.2s ease;
            }}

            summary:hover {{
                background-color: #f1f5f9;
            }}

            details[open] summary {{
                border-bottom: 1px solid var(--border);
                background-color: #f1f5f9;
            }}

            .details-content {{
                padding: 20px;
            }}

            /* Interpretations list */
            .interpretations {{
                background-color: #fffbeb;
                padding: 20px;
                border-left: 5px solid #d97706;
                border-radius: var(--radius-md);
                margin-bottom: 30px;
            }}

            .interpretations ul {{
                padding-left: 20px;
                margin: 10px 0 0 0;
            }}

            .interpretations li {{
                margin-bottom: 8px;
            }}

            /* Graphics layout */
            .grid-2 {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}

            .img-container {{
                background: #ffffff;
                border: 1px solid var(--border);
                border-radius: var(--radius-md);
                padding: 16px;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }}

            .img-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 4px;
            }}

            .img-title {{
                margin-top: 10px;
                font-weight: 600;
                font-family: 'Outfit', sans-serif;
                color: var(--text-secondary);
            }}

            @media print {{
                body {{
                    background-color: #ffffff;
                    padding: 0;
                }}
                .container {{
                    box-shadow: none;
                    border: none;
                    padding: 0;
                    max-width: 100%;
                }}
                details {{
                    border: none;
                    margin-bottom: 30px;
                    page-break-inside: avoid;
                }}
                details summary {{
                    display: block;
                    font-size: 1.3rem;
                    border-bottom: 2px solid #333;
                    background: none;
                    padding: 0 0 5px 0;
                }}
                .details-content {{
                    padding: 10px 0;
                }}
                .grid-2 {{
                    display: block;
                }}
                .img-container {{
                    page-break-inside: avoid;
                    margin-bottom: 30px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatório de Análise - Eletrofisiologia Vegetal</h1>

            <div class="section experiment-desc">
                <h2>Descrição do Experimento</h2>
                <p>{experiment_description}</p>
            </div>

            {highlight_html}

            {interpretations_html}

            <div class="section">
                <h2>Tabelas de Comparação</h2>
                <h3>Resumo das Médias por Classe (Bruto e Z-Score)</h3>
                {summary_df_transposed.to_html(float_format='%.4f')}
                <h3>Diferença Percentual (Relativo Bruto)</h3>
                {comparison_df_transposed.to_html(float_format='%.2f')}
                {ols_html}
            </div>

            <div class="section">
                <h2>Gráficos das Características</h2>
                {graphs_html}
            </div>
        </div>
    </body>
    </html>
    """

    report_path = results_path / "report.html"
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(html)

    gui_log_callback(f"Relatório gerado em {report_path}")
    return report_path