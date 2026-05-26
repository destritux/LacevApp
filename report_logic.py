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
    json_data_str = "{}"
    
    try:
        gui_log_callback("Tentando processar estatísticas de Regressão Linear (OLS) pareadas...")
        import statsmodels.formula.api as smf
        import numpy as np
        import itertools
        import json
        
        master_df = pd.concat(master_df_list, ignore_index=True)
        z_cols = [c for c in master_df.columns if c.endswith('_Z')]
        classes = sorted(all_data.keys())
        
        # Serializar dados para o Dashboard Interativo
        embedded_data = {}
        for c_name, c_df in all_data.items():
            # Substitui NaN por None para que o JSON serializado use null de forma segura
            c_df_clean = c_df.copy()
            c_df_clean = c_df_clean.replace({np.nan: None})
            embedded_data[c_name] = c_df_clean.to_dict(orient='records')
        json_data_str = json.dumps(embedded_data)
        
        def _get_metric_url(metric_name):
            base_name = metric_name.replace('_Z', '')
            urls = {
                'Raw_mean': 'https://en.wikipedia.org/wiki/Mean',
                'Raw_min': 'https://en.wikipedia.org/wiki/Sample_maximum_and_minimum',
                'Raw_max': 'https://en.wikipedia.org/wiki/Sample_maximum_and_minimum',
                'Raw_var': 'https://en.wikipedia.org/wiki/Variance',
                'ApEn': 'https://en.wikipedia.org/wiki/Approximate_entropy',
                'SampleEntropy': 'https://en.wikipedia.org/wiki/Sample_entropy',
                'DFA': 'https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis',
                'Lyap_r': 'https://en.wikipedia.org/wiki/Lyapunov_exponent',
                'Lyap_e': 'https://en.wikipedia.org/wiki/Lyapunov_exponent',
                'FFT_mean': 'https://en.wikipedia.org/wiki/Fast_Fourier_transform',
                'FFT_min': 'https://en.wikipedia.org/wiki/Fast_Fourier_transform',
                'FFT_max': 'https://en.wikipedia.org/wiki/Fast_Fourier_transform',
                'FFT_var': 'https://en.wikipedia.org/wiki/Fast_Fourier_transform',
                'PSD_mean': 'https://en.wikipedia.org/wiki/Spectral_density',
                'PSD_min': 'https://en.wikipedia.org/wiki/Spectral_density',
                'PSD_max': 'https://en.wikipedia.org/wiki/Spectral_density',
                'PSD_var': 'https://en.wikipedia.org/wiki/Spectral_density',
                'Bandpower_ISO': 'https://pubmed.ncbi.nlm.nih.gov/?term=infra-slow+oscillations+plants',
                'Bandpower_Delta': 'https://en.wikipedia.org/wiki/Delta_wave',
                'Bandpower_Theta': 'https://en.wikipedia.org/wiki/Theta_wave',
                'Bandpower_Alpha': 'https://en.wikipedia.org/wiki/Alpha_wave',
                'Bandpower_Beta': 'https://en.wikipedia.org/wiki/Beta_wave'
            }
            return urls.get(base_name, 'https://en.wikipedia.org/wiki/Spectral_analysis')

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
                    <p>O par de classes com <b>maior diferença geral</b> é <b>{most_diff_pair}</b>, apresentando <b>{most_diff_count}</b> características normalizadas com diferença estatisticamente significativa (p < 0.05).</p>
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
                    url = _get_metric_url(r['Métrica Z-Score'])
                    metric_link = f"<a href='{url}' target='_blank' class='info-link'><b>{r['Métrica Z-Score']}</b> ℹ️</a>"
                    table_rows += f"<tr><td>{metric_link}</td><td>{coef_str}</td><td>{p_str}</td><td>{badge}</td></tr>"
                
                pair_table = f"""
                <table>
                    <thead>
                        <tr>
                            <th>Métrica Z-Score (Definição ℹ️)</th>
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

    gui_log_callback("Construindo HTML final...")

    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LacevApp - Painel Científico de Eletrofisiologia</title>
    <!-- Carrega fontes modernas e Chart.js via CDN -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
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
            --radius-lg: 12px;
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
            --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }

        body.dark-theme {
            --bg-main: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --border: #334155;
            --primary-light: #064e3b;
            --primary-dark: #34d399;
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.3);
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            line-height: 1.5;
            margin: 0;
            padding: 20px;
            transition: var(--transition);
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
            background: var(--bg-card);
            padding: 40px;
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow);
            border: 1px solid var(--border);
            transition: var(--transition);
        }

        /* Top Header */
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
            color: white;
            padding: 30px 40px;
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow);
            margin-bottom: 30px;
        }

        .header-left h1 {
            margin: 0;
            font-family: 'Outfit', sans-serif;
            font-size: 2.1rem;
            font-weight: 800;
            color: #ffffff;
        }

        .header-left p {
            margin: 6px 0 0 0;
            color: #a7f3d0;
            font-size: 1.05rem;
            font-weight: 300;
        }

        .theme-toggle-btn {
            background-color: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            padding: 10px 18px;
            border-radius: var(--radius-md);
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
        }

        .theme-toggle-btn:hover {
            background-color: rgba(255, 255, 255, 0.22);
            transform: translateY(-1px);
        }

        h2, h3, h4 {
            font-family: 'Outfit', sans-serif;
            color: #064e3b;
        }

        body.dark-theme h2, body.dark-theme h3, body.dark-theme h4 {
            color: #34d399;
        }

        /* Tabs System */
        .tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 25px;
            border-bottom: 2px solid var(--border);
            padding-bottom: 8px;
        }

        .tab-btn {
            background: none;
            border: none;
            padding: 12px 22px;
            font-family: 'Outfit', sans-serif;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            color: var(--text-secondary);
            border-radius: var(--radius-md);
            transition: var(--transition);
        }

        .tab-btn:hover {
            background-color: var(--bg-main);
            color: var(--text-primary);
        }

        .tab-btn.active {
            background-color: var(--primary-light);
            color: var(--primary-dark);
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Section Layouts */
        .section {
            margin-bottom: 35px;
        }

        .section.experiment-desc {
            background-color: var(--bg-main);
            padding: 24px;
            border-left: 5px solid var(--primary);
            border-radius: var(--radius-md);
            margin-bottom: 30px;
        }

        .section.experiment-desc h2 {
            margin-top: 0;
            font-size: 1.4rem;
        }

        .highlight-box {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border-left: 5px solid var(--primary);
            border-radius: var(--radius-md);
            padding: 22px;
            margin-bottom: 30px;
            color: #064e3b;
        }

        body.dark-theme .highlight-box {
            background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
            border-color: #34d399;
            color: #a7f3d0;
        }

        .highlight-box h3 {
            margin-top: 0;
            font-size: 1.2rem;
            color: inherit;
        }

        .highlight-box p {
            margin: 6px 0;
            font-size: 1.05rem;
        }

        /* Interactive Dashboard Styling */
        .control-panel {
            background-color: var(--bg-main);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 24px;
            margin-bottom: 25px;
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
            align-items: center;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .control-group label {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--text-secondary);
        }

        select {
            padding: 10px 14px;
            font-family: 'Inter', sans-serif;
            font-size: 0.95rem;
            border-radius: 6px;
            border: 1px solid var(--border);
            background-color: var(--bg-card);
            color: var(--text-primary);
            outline: none;
            min-width: 300px;
            cursor: pointer;
            transition: var(--transition);
        }

        select:focus {
            border-color: var(--primary);
        }

        .toggle-container {
            display: flex;
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 4px;
        }

        .toggle-btn {
            background: none;
            border: none;
            padding: 8px 16px;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            color: var(--text-secondary);
            border-radius: 4px;
            transition: var(--transition);
        }

        .toggle-btn.active {
            background-color: var(--primary);
            color: white;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 24px;
        }

        .chart-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 24px;
            min-height: 400px;
            display: flex;
            flex-direction: column;
        }

        .chart-card h3 {
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.15rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 10px;
        }

        .chart-container {
            position: relative;
            flex-grow: 1;
            height: 320px;
            width: 100%;
        }

        /* Beautiful Tables */
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 30px;
            font-size: 0.92rem;
        }

        th, td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }

        th {
            background-color: var(--bg-main);
            color: var(--text-secondary);
            font-weight: 600;
        }

        tr:hover td {
            background-color: var(--bg-main);
        }

        .info-link {
            color: var(--primary-dark);
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            transition: var(--transition);
        }

        .info-link:hover {
            color: var(--secondary);
            text-decoration: underline;
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-success {
            background-color: var(--primary-light);
            color: var(--primary-dark);
        }

        body.dark-theme .badge-success {
            background-color: #064e3b;
            color: #34d399;
        }

        .badge-neutral {
            background-color: var(--bg-main);
            color: var(--text-secondary);
        }

        /* Details / Accordion */
        details {
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            margin-bottom: 15px;
            background-color: var(--bg-card);
            overflow: hidden;
            transition: var(--transition);
        }

        summary {
            padding: 14px 20px;
            font-weight: 600;
            font-family: 'Outfit', sans-serif;
            cursor: pointer;
            background-color: var(--bg-main);
            user-select: none;
            transition: var(--transition);
        }

        summary:hover {
            background-color: var(--border);
        }

        details[open] summary {
            border-bottom: 1px solid var(--border);
            background-color: var(--border);
        }

        .details-content {
            padding: 24px;
            background-color: var(--bg-card);
        }

        /* Interpretations list */
        .interpretations {
            background-color: #fffbeb;
            padding: 24px;
            border-left: 5px solid #d97706;
            border-radius: var(--radius-md);
            margin-bottom: 30px;
            color: #78350f;
        }

        body.dark-theme .interpretations {
            background-color: #78350f20;
            border-color: #d97706;
            color: #fef3c7;
        }

        .interpretations h2 {
            margin-top: 0;
            font-size: 1.3rem;
            color: inherit;
        }

        .interpretations ul {
            padding-left: 20px;
            margin: 10px 0 0 0;
        }

        .interpretations li {
            margin-bottom: 8px;
        }

        /* Glossary Scientific cards */
        .glossary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }

        .glossary-card {
            background-color: var(--bg-main);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 24px;
            transition: var(--transition);
        }

        .glossary-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow);
            border-color: var(--primary);
        }

        .glossary-card h4 {
            margin-top: 0;
            font-size: 1.15rem;
            margin-bottom: 10px;
        }

        .glossary-card p {
            font-size: 0.92rem;
            color: var(--text-secondary);
            margin-bottom: 16px;
        }

        .glossary-card a {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--primary);
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }

        .glossary-card a:hover {
            text-decoration: underline;
        }

        @media print {
            body {
                background-color: #ffffff;
                padding: 0;
            }
            .container {
                box-shadow: none;
                border: none;
                padding: 0;
                max-width: 100%;
            }
            .theme-toggle-btn, .tabs, .control-panel {
                display: none !important;
            }
            .tab-content {
                display: block !important;
                margin-bottom: 40px;
                page-break-after: always;
            }
            details {
                border: none;
                margin-bottom: 35px;
                page-break-inside: avoid;
            }
            details summary {
                display: block;
                font-size: 1.3rem;
                border-bottom: 2px solid #333;
                background: none;
                padding: 0 0 5px 0;
            }
            .details-content {
                padding: 10px 0;
            }
            .grid-2 {
                display: block;
            }
            .chart-card {
                page-break-inside: avoid;
                margin-bottom: 30px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        
        <!-- Header Section -->
        <div class="header-container">
            <div class="header-left">
                <h1>🌱 LacevApp — Eletrofisiologia Vegetal</h1>
                <p>Relatório Científico & Comparativos Estatísticos</p>
            </div>
            <button class="theme-toggle-btn" onclick="toggleTheme()">🌓 Alternar Tema</button>
        </div>

        <!-- System Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('dashboard')">📈 Dashboard Interativo</button>
            <button class="tab-btn" onclick="switchTab('resumos')">📊 Médias & Comparativos</button>
            <button class="tab-btn" onclick="switchTab('ols')">🔬 Regressão Pareada OLS</button>
            <button class="tab-btn" onclick="switchTab('glossario')">📖 Glossário & Informação</button>
        </div>

        <!-- Experiment description (Global) -->
        <div class="section experiment-desc">
            <h2>Descrição do Experimento</h2>
            <p>__EXPERIMENT_DESCRIPTION__</p>
        </div>

        <!-- Highlight Box (Global) -->
        __HIGHLIGHT_HTML__

        <!-- Tab 1: Dashboard Interativo -->
        <div id="tab-dashboard" class="tab-content active">
            <div class="control-panel">
                <div class="control-group">
                    <label for="metric-select">🔍 Selecione a Métrica Fisiológica:</label>
                    <select id="metric-select" onchange="updateCharts()">
                        <optgroup label="Medidas Temporais (Voltagem)">
                            <option value="Raw_mean">Média de Voltagem Bruta</option>
                            <option value="Raw_min">Mínimo de Voltagem Bruta</option>
                            <option value="Raw_max">Máximo de Voltagem Bruta</option>
                            <option value="Raw_var">Variância de Voltagem Bruta</option>
                        </optgroup>
                        <optgroup label="Complexidade Dinâmica e Caos">
                            <option value="ApEn">Approximate Entropy (ApEn)</option>
                            <option value="SampleEntropy" selected>Sample Entropy (SampEn)</option>
                            <option value="DFA">Detrended Fluctuation Analysis (Expoente de Hurst)</option>
                            <option value="Lyap_r">Expoente de Lyapunov (Dimensão ACF)</option>
                            <option value="Lyap_e">Expoente de Lyapunov (Dimensão FNN)</option>
                        </optgroup>
                        <optgroup label="Frequência e Espectro">
                            <option value="FFT_mean">Média de FFT</option>
                            <option value="FFT_min">Mínimo de FFT</option>
                            <option value="FFT_max">Máximo de FFT</option>
                            <option value="FFT_var">Variância de FFT</option>
                            <option value="PSD_mean">Média de PSD (Multitaper)</option>
                            <option value="PSD_min">Mínimo de PSD (Multitaper)</option>
                            <option value="PSD_max">Máximo de PSD (Multitaper)</option>
                            <option value="PSD_var">Variância de PSD (Multitaper)</option>
                        </optgroup>
                        <optgroup label="Bandas de Potência Fisiológicas">
                            <option value="Bandpower_ISO">Oscilações Infra-Lentas (ISO, 0.005-0.1 Hz)</option>
                            <option value="Bandpower_Delta">Banda Delta (0.1-4 Hz)</option>
                            <option value="Bandpower_Theta">Banda Theta (4-8 Hz)</option>
                            <option value="Bandpower_Alpha">Banda Alpha (8-12 Hz)</option>
                            <option value="Bandpower_Beta">Banda Beta (12-30 Hz)</option>
                        </optgroup>
                    </select>
                </div>
                
                <div class="control-group">
                    <label>📊 Tipo de Escala:</label>
                    <div class="toggle-container">
                        <button id="toggle-raw" class="toggle-btn active" onclick="setMode('raw')">Valores Brutos</button>
                        <button id="toggle-zscore" class="toggle-btn" onclick="setMode('zscore')">Z-Scores (Baseline)</button>
                    </div>
                </div>
            </div>

            <div class="grid-2">
                <div class="chart-card">
                    <h3>Evolução Temporal (Minuto a Minuto)</h3>
                    <div class="chart-container">
                        <canvas id="lineChart"></canvas>
                    </div>
                </div>
                
                <div class="chart-card">
                    <h3>Comparação de Médias Gerais</h3>
                    <div class="chart-container">
                        <canvas id="barChart"></canvas>
                    </div>
                </div>
            </div>
            
            <p style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 15px; text-align: center;">
                💡 <i>Dica: Clique nas classes na legenda do gráfico de linhas para ocultar/exibir curvas específicas.</i>
            </p>
        </div>

        <!-- Tab 2: Resumos & Comparativos -->
        <div id="tab-resumos" class="tab-content">
            __INTERPRETATIONS_HTML__
            
            <div class="section">
                <h2>Resumo das Médias por Classe (Bruto e Z-Score)</h2>
                __SUMMARY_TABLE__
            </div>

            <div class="section">
                <h2>Diferença Percentual (Relativo Bruto)</h2>
                __COMPARISON_TABLE__
            </div>
        </div>

        <!-- Tab 3: OLS Pareado -->
        <div id="tab-ols" class="tab-content">
            <div class="section">
                <h2>Análise de Regressão Linear Pareada (OLS)</h2>
                <p>O modelo ajustado é do tipo <code>Feature_Z ~ C(TreatmentClass)</code> aplicado isoladamente para cada combinação possível de par de tratamento, permitindo identificar diferenças específicas localizadas entre cada par de classes.</p>
                __OLS_HTML__
            </div>
        </div>

        <!-- Tab 4: Glossário Científico -->
        <div id="tab-glossario" class="tab-content">
            <h2>Glossário Científico das Métricas</h2>
            <p style="margin-bottom: 25px;">Consulte abaixo a base teórica de cada métrica utilizada nos cálculos eletrofisiológicos do LacevApp, com links para aprofundamento acadêmico.</p>
            
            <div class="glossary-grid">
                <div class="glossary-card">
                    <h4>Approximate Entropy (ApEn)</h4>
                    <p>Mede a regularidade e flutuações de ruído em séries temporais. Valores menores indicam sinais muito repetitivos e estruturados, enquanto valores maiores indicam maior complexidade ou imprevisibilidade.</p>
                    <a href="https://en.wikipedia.org/wiki/Approximate_entropy" target="_blank">Ler mais na Wikipédia ↗</a>
                </div>

                <div class="glossary-card">
                    <h4>Sample Entropy (SampEn)</h4>
                    <p>Uma evolução direta da Approximate Entropy (ApEn) projetada para eliminar o viés de auto-comparação (auto-matching). SampEn exibe estabilidade matemática e consistência estatística mesmo em fragmentos de sinal mais curtos.</p>
                    <a href="https://en.wikipedia.org/wiki/Sample_entropy" target="_blank">Ler mais na Wikipédia ↗</a>
                </div>

                <div class="glossary-card">
                    <h4>Detrended Fluctuation Analysis (DFA)</h4>
                    <p>Mapeia correlações de longo alcance em sinais biológicos não estacionários. O expoente de escala estimado (expoente de Hurst) indica a presença de memória fractal de longo prazo no comportamento bioelétrico da planta.</p>
                    <a href="https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis" target="_blank">Ler mais na Wikipédia ↗</a>
                </div>

                <div class="glossary-card">
                    <h4>Expoente de Lyapunov</h4>
                    <p>Parâmetro central da Teoria do Caos que quantifica o grau de caoticidade de um sistema dinâmico. Mede a taxa de divergência exponencial de trajetórias inicialmente próximas no espaço de fases.</p>
                    <a href="https://en.wikipedia.org/wiki/Lyapunov_exponent" target="_blank">Ler mais na Wikipédia ↗</a>
                </div>

                <div class="glossary-card">
                    <h4>Densidade Espectral (PSD - Multitaper)</h4>
                    <p>Estimação de alta resolução baseada no estimador Multitaper (DPSS). Minimiza vazamentos espectrais de energia e reduz a variância do espectro de potência através de tapers ortogonais ótimos de Slepian.</p>
                    <a href="https://en.wikipedia.org/wiki/Spectral_density" target="_blank">Ler mais na Wikipédia ↗</a>
                </div>

                <div class="glossary-card">
                    <h4>Oscilações Infra-Lentas (ISO)</h4>
                    <p>Banda eletrofisiológica vegetal na faixa de 0.005 Hz a 0.1 Hz. Corresponde a ritmos bioelétricos lentos fundamentais para a regulação do balanço hídrico, fluxo vascular e sinalização sistêmica sob estresse.</p>
                    <a href="https://pubmed.ncbi.nlm.nih.gov/?term=infra-slow+oscillations+plants" target="_blank">Buscar artigos no PubMed ↗</a>
                </div>
            </div>
        </div>

    </div>

    <!-- Script de Configuração dos Gráficos Interativos -->
    <script>
        // Dados embutidos serializados pelo Python
        const dataset = __DATASET_JSON__;

        let currentMode = 'raw';
        let lineChart = null;
        let barChart = null;

        // Cores premium para as classes
        const classColors = {
            'C1': '#10b981', // Emerald
            'C2': '#06b6d4', // Teal
            'C3': '#6366f1', // Indigo
            'C4': '#8b5cf6', // Purple
            'C5': '#f59e0b', // Amber
            'C6': '#f43f5e'  // Rose
        };

        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            
            document.getElementById('tab-' + tabId).classList.add('active');
            
            const clickedBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
            if (clickedBtn) clickedBtn.classList.add('active');
        }

        function toggleTheme() {
            document.body.classList.toggle('dark-theme');
            updateCharts();
        }

        function setMode(mode) {
            currentMode = mode;
            document.getElementById('toggle-raw').classList.toggle('active', mode === 'raw');
            document.getElementById('toggle-zscore').classList.toggle('active', mode === 'zscore');
            updateCharts();
        }

        function updateCharts() {
            const metricSelect = document.getElementById('metric-select');
            const metric = metricSelect.value;
            
            let dataKey = currentMode === 'zscore' ? metric + '_Z' : metric;
            
            // Garantia caso a métrica não tenha correspondente Z-Score
            const sampleClass = Object.keys(dataset)[0];
            const sampleData = dataset[sampleClass];
            if (currentMode === 'zscore' && sampleData && sampleData.length > 0 && !(dataKey in sampleData[0])) {
                dataKey = metric;
            }
            
            const labelText = metricSelect.options[metricSelect.selectedIndex].text;
            const yLabel = currentMode === 'zscore' ? `${labelText} (Z-Score)` : labelText;
            
            // Labels de Eixo X (Minutos)
            let maxLen = 0;
            Object.keys(dataset).forEach(c => {
                if (dataset[c].length > maxLen) maxLen = dataset[c].length;
            });
            const labels = Array.from({length: maxLen}, (_, i) => `${i+1} min`);
            
            const lineDatasets = [];
            const barLabels = [];
            const barDataValues = [];
            const barBackgrounds = [];
            const barBorders = [];
            
            const isDark = document.body.classList.contains('dark-theme');
            const gridColor = isDark ? '#334155' : '#e2e8f0';
            const textColor = isDark ? '#cbd5e1' : '#475569';
            
            Object.keys(dataset).sort().forEach(className => {
                const classData = dataset[className];
                const values = classData.map(row => row[dataKey]);
                
                lineDatasets.push({
                    label: className,
                    data: values,
                    borderColor: classColors[className] || '#cbd5e1',
                    backgroundColor: (classColors[className] || '#cbd5e1') + '10',
                    borderWidth: 2.5,
                    tension: 0.25,
                    fill: false
                });
                
                // Média geral para barra
                const validValues = values.filter(v => v !== null && v !== undefined);
                const avg = validValues.length > 0 ? validValues.reduce((a, b) => a + b, 0) / validValues.length : 0;
                
                barLabels.push(className);
                barDataValues.push(avg);
                barBackgrounds.push((classColors[className] || '#cbd5e1') + '30');
                barBorders.push(classColors[className] || '#cbd5e1');
            });
            
            if (lineChart) lineChart.destroy();
            if (barChart) barChart.destroy();
            
            // Gráfico de Linhas
            const ctxLine = document.getElementById('lineChart').getContext('2d');
            lineChart = new Chart(ctxLine, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: lineDatasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: { color: textColor, font: { family: 'Inter', weight: 500 } }
                        },
                        tooltip: {
                            padding: 12,
                            boxPadding: 6,
                            titleFont: { family: 'Outfit', size: 13 },
                            bodyFont: { family: 'Inter' }
                        }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: { color: textColor, font: { family: 'Inter' } }
                        },
                        y: {
                            title: { display: true, text: yLabel, color: textColor, font: { family: 'Outfit', weight: 600 } },
                            grid: { color: gridColor },
                            ticks: { color: textColor, font: { family: 'Inter' } }
                        }
                    }
                }
            });
            
            // Gráfico de Barras
            const ctxBar = document.getElementById('barChart').getContext('2d');
            barChart = new Chart(ctxBar, {
                type: 'bar',
                data: {
                    labels: barLabels,
                    datasets: [{
                        data: barDataValues,
                        backgroundColor: barBackgrounds,
                        borderColor: barBorders,
                        borderWidth: 1.5,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            padding: 12,
                            titleFont: { family: 'Outfit', size: 13 },
                            bodyFont: { family: 'Inter' }
                        }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: { color: textColor, font: { family: 'Inter' } }
                        },
                        y: {
                            title: { display: true, text: `Média de ${yLabel}`, color: textColor, font: { family: 'Outfit', weight: 600 } },
                            grid: { color: gridColor },
                            ticks: { color: textColor, font: { family: 'Inter' } }
                        }
                    }
                }
            });
        }

        // Inicializar painéis de gráficos
        window.onload = function() {
            updateCharts();
        };
    </script>
</body>
</html>
"""

    # Realiza as substituições no template HTML de forma robusta e livre de problemas com chaves do f-string
    html = html_template
    html = html.replace("__EXPERIMENT_DESCRIPTION__", str(experiment_description))
    html = html.replace("__HIGHLIGHT_HTML__", str(highlight_html))
    html = html.replace("__INTERPRETATIONS_HTML__", str(interpretations_html))
    html = html.replace("__SUMMARY_TABLE__", summary_df_transposed.to_html(float_format='%.4f'))
    html = html.replace("__COMPARISON_TABLE__", comparison_df_transposed.to_html(float_format='%.2f'))
    html = html.replace("__OLS_HTML__", str(ols_html))
    html = html.replace("__DATASET_JSON__", json_data_str)

    report_path = results_path / "report.html"
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(html)

    gui_log_callback(f"Relatório gerado em {report_path}")
    return report_path