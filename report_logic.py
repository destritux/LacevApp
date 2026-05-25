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

    # --- MODELO LINEAR (OLS) COM STATSMODELS ---
    ols_html = ""
    try:
        gui_log_callback("Tentando processar estatísticas de Regressão Linear (OLS)...")
        import statsmodels.formula.api as smf
        import numpy as np
        master_df = pd.concat(master_df_list, ignore_index=True)
        
        ols_results = []
        # Analisa apenas variáveis normalizadas Z-Score (Extrapolação para Espécie)
        z_cols = [c for c in master_df.columns if c.endswith('_Z')]
        
        if z_cols:
            for z_col in z_cols:
                # Trata caracteres inválidos para a fórmula do statsmodels
                safe_col = z_col.replace('-', '_')
                temp_df = master_df.rename(columns={z_col: safe_col})
                temp_df = temp_df.dropna(subset=[safe_col, 'TreatmentClass'])
                
                try:
                    # Fórmula OLS clássica
                    formula = f"{safe_col} ~ C(TreatmentClass)"
                    md = smf.ols(formula, temp_df)
                    mdf = md.fit()
                    
                    # Extrai o primeiro termo de tratamento em relação à referência
                    p_value = mdf.pvalues.iloc[1] if len(mdf.pvalues) > 1 else np.nan
                    coef = mdf.params.iloc[1] if len(mdf.params) > 1 else np.nan
                    
                    sig = "Sim (p<0.05)" if p_value < 0.05 else "Não"
                    ols_results.append({
                        'Métrica Z-Score': z_col,
                        'Efeito (Coef)': coef,
                        'P-Value': p_value,
                        'Significativo?': sig
                    })
                except Exception as e:
                    gui_log_callback(f"Aviso no cálculo OLS de {z_col}: {e}")
            
            if ols_results:
                ols_df = pd.DataFrame(ols_results)
                # Salva estatísticas de regressão na pasta 'results'
                ols_df.to_csv(results_path / "regression_stats.csv", index=False)
                ols_html = f"<h3>Análise Estatística de Regressão (OLS via Statsmodels)</h3>{ols_df.to_html(float_format='%.4f', index=False)}"
            else:
                ols_html = "<p><i>Modelo OLS não gerou resultados.</i></p>"
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
        graphs_html = ''.join([
            f'<h3>{Path(g).name.replace("_light.png", "").replace("_dark.png", "").replace("TDAF_", "")}</h3>'
            f'<img src="../graphics/{Path(g).name}" alt="Gráfico de {Path(g).stem}">'
            for g in graph_files
        ])
    else:
        graphs_html = "<p><i>Nenhum gráfico disponível.</i></p>"

    html = f"""
    <html>
    <head>
        <title>Relatório de Análise Eletrofisiológica</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f4f4f9; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #bdc3c7; padding-bottom: 10px; }}
            .container {{ width: 210mm; min-height: 297mm; margin: 20px auto; background: white; padding: 20mm;
                          border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 25px; font-size: 9pt; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #ecf0f1; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            img {{ max-width: 100%; height: auto; display: block; margin: 20px 0;
                   border: 1px solid #ddd; border-radius: 4px; }}
            .section {{ margin-bottom: 45px; }}
            .experiment-desc {{ background-color: #eaf2f8; padding: 20px; border-left: 5px solid #3498db; border-radius: 5px; }}
            .interpretations {{ background-color: #fcf3cf; padding: 20px; border-left: 5px solid #f1c40f; border-radius: 5px; }}
            .interpretations ul {{ padding-left: 20px; }}
            .interpretations li {{ margin-bottom: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatório de Análise - Eletrofisiologia Vegetal</h1>

            <div class="section experiment-desc">
                <h2>Descrição do Experimento</h2>
                <p>{experiment_description}</p>
            </div>

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