"""
LacevApp - Aplicação para processamento e análise de dados de microvoltagem.

Este módulo implementa uma interface gráfica para processamento e análise de dados
de microvoltagem, permitindo a visualização e análise de diferentes tipos de gráficos.
"""

import warnings
import os
import glob
import pandas as pd
from typing import List, Dict, Any, Optional
from tkinter import OptionMenu, StringVar, ttk, filedialog, messagebox
import tkinter as tk
from PIL import Image, ImageTk
from ttkthemes import ThemedTk

from utius import lacevApp, Filter, Features

# Desabilitar avisos para melhor experiência do usuário
warnings.filterwarnings('ignore')

class LacevAppGUI:
    """Classe principal que gerencia a interface gráfica da aplicação."""
    
    def __init__(self):
        """Inicializa a interface gráfica da aplicação."""
        self.root = ThemedTk(theme='darkly')
        self.root.title('lacevApp version=alpha0.2')
        self.root.iconbitmap('lacevapp.ico')
        
        self.setup_variables()
        self.create_widgets()
        
    def setup_variables(self) -> None:
        """Configura as variáveis da interface."""
        self.clicked = StringVar(value='Ploats')
        self.inst_text = tk.StringVar()
        self.inst_text.set("lacevApp needs a lot of processing power. Therefore, it is highly recommended that you close all other programs!")
        self.browse_text = tk.StringVar(value='Open Folder')
        
    def create_widgets(self) -> None:
        """Cria e posiciona todos os widgets da interface."""
        # Menu dropdown
        self.drop = OptionMenu(
            self.root, 
            self.clicked, 
            "FFT-PSD", 
            "TDAF", 
            "TDAFLog", 
            command=self.action_function
        )
        self.drop.config(font="Raleway", bg='green', fg='white', height=2, width=15)
        self.drop.grid(column=0, row=0)
        
        # Logo
        self.setup_logo()
        
        # Barra de progresso
        self.prog_bar = ttk.Progressbar(
            self.root,
            orient='horizontal',
            mode='determinate',
            length=500,
            maximum=100
        )
        self.prog_bar.grid(column=1, row=1)
        
        # Texto de instrução
        instr = tk.Label(
            self.root,
            textvariable=self.inst_text,
            font="Raleway",
            bg='red',
            fg='white'
        )
        instr.grid(columnspan=2, column=0, row=2)
        
        # Botão de navegação
        self.browse_bin = tk.Button(
            self.root,
            textvariable=self.browse_text,
            command=self.open_dir,
            font="Raleway",
            bg='green',
            fg='white',
            height=2,
            width=15
        )
        self.browse_bin.grid(column=2, row=2)
        
    def setup_logo(self) -> None:
        """Configura e exibe o logo da aplicação."""
        try:
            logo = Image.open('logo.png')
            logo = ImageTk.PhotoImage(logo)
            logo_label = tk.Label(image=logo)
            logo_label.image = logo
            logo_label.grid(column=1, row=0)
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar o logo: {str(e)}")
            
    def open_dir(self) -> None:
        """Abre um diretório e inicia o processamento dos dados."""
        self.inst_text.set("loading...")
        self.browse_bin.destroy()
        
        self.root.directory = filedialog.askdirectory()
        if not self.root.directory:
            return
            
        self.process_directory()
        
    def process_directory(self) -> None:
        """Processa os arquivos do diretório selecionado."""
        try:
            l_app = lacevApp()
            l_app.startLacevApp(self.root.directory)
            
            self.setup_text_box(l_app.startDir())
            self.process_files(l_app)
            l_app.explore(l_app.getGfilePathFe())
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar diretório: {str(e)}")
            
    def setup_text_box(self, text_messages: List[str]) -> None:
        """Configura a caixa de texto com scrollbar."""
        li_frame = ttk.Frame(self.root, width=40, height=10)
        li_frame.grid(column=1, row=2, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(li_frame)
        self.text_box = tk.Text(li_frame, height=10, width=50, padx=15, pady=15)
        self.text_box.tag_configure("center", justify="center")
        
        length_text = 100/(len(text_messages)-1) if text_messages else 0
        
        for message in text_messages:
            self.text_box.insert(1.0, message)
            self.prog_bar['value'] += length_text
            self.prog_bar.update_idletasks()
            
        self.text_box.pack()
        scrollbar.configure(command=self.text_box.yview)
        scrollbar.pack(side="right", fill="y")
        
        self.prog_bar['value'] = 0
        self.prog_bar.update_idletasks()
        
    def process_files(self, l_app: lacevApp) -> None:
        """Processa os arquivos do diretório."""
        filter_obj = Filter()
        features_obj = Features()
        classe = 0
        minu_cut = 3
        
        for path_c in l_app.getfilepathC():
            self.process_class_files(l_app, path_c, classe, minu_cut, filter_obj, features_obj)
            classe += 1
            
    def process_class_files(self, l_app: lacevApp, path_c: str, classe: int, 
                          minu_cut: int, filter_obj: Filter, features_obj: Features) -> None:
        """Processa os arquivos de uma classe específica."""
        path_fc = os.path.join(l_app.getfilepatR(), path_c)
        directory = os.listdir(path_fc)
        resu = []
        
        for file in directory:
            file_n, exte = os.path.splitext(str(file))
            
            if exte in ['.csv', '.txt']:
                self.process_single_file(
                    l_app, path_fc, file, file_n, path_c, classe,
                    minu_cut, filter_obj, features_obj, resu
                )
                
        self.save_class_features(l_app, path_c, classe, resu)
        
    def process_single_file(self, l_app: lacevApp, path_fc: str, file: str,
                          file_n: str, path_c: str, classe: int, minu_cut: int,
                          filter_obj: Filter, features_obj: Features, resu: List) -> None:
        """Processa um único arquivo."""
        try:
            path_file_final = os.path.join(path_fc, file)
            data_p = pd.read_csv(
                path_file_final,
                encoding='utf-8',
                delimiter=",",
                engine='c',
                low_memory=False,
                memory_map=True,
            )
            
            # Verifica se o DataFrame está vazio
            if data_p.empty:
                print(f"Arquivo vazio: {file_n}")
                return
                
            # Verifica se há dados suficientes para processar
            if len(data_p.columns) == 0:
                print(f"Arquivo sem colunas: {file_n}")
                return
                
            # Verifica se há dados na primeira coluna
            if len(data_p.iloc[:, 0]) == 0:
                print(f"Sem dados na primeira coluna: {file_n}")
                return
            
            # Processar dados
            try:
                m_volt = filter_obj.starFilter(data_p.to_numpy()[:,0].ravel(), pathC=path_c, fileN=file_n)
                if m_volt is None or len(m_volt) < 2:
                    print(f"Erro no processamento do filtro para: {file_n}")
                    return
                    
                data_p['MicroVF'] = m_volt[1]
            except IndexError as e:
                print(f"Erro de índice ao processar filtro para {file_n}: {str(e)}")
                return
            
            # Processar tempo
            try:
                time = self.process_time(data_p, minu_cut)
            except Exception as e:
                print(f"Erro ao processar tempo para {file_n}: {str(e)}")
                time = 0
            
            # Processar data
            try:
                date = self.process_date(data_p)
            except Exception as e:
                print(f"Erro ao processar data para {file_n}: {str(e)}")
                date = []
            
            # Calcular features
            try:
                resu = features_obj.startFeature(
                    m_volt[1], minu_cut, time, date, classe, path_c, 60, resu,
                    self.text_box, file_n, self.prog_bar
                )
            except Exception as e:
                print(f"Erro ao calcular features para {file_n}: {str(e)}")
                return
            
            # Salvar resultados
            try:
                path_saved = os.path.join(str(path_c), str(file_n))
                df = features_obj.salveDf(resu, path_saved, mode=1)
                
                if isinstance(df, FileNotFoundError):
                    print(f'Error saving file: {str(df)}')
            except Exception as e:
                print(f"Erro ao salvar resultados para {file_n}: {str(e)}")
                
        except Exception as e:
            error_msg = f"Erro ao processar arquivo {file_n}: {str(e)}"
            print(error_msg)
            messagebox.showerror("Erro", error_msg)
            
    def process_time(self, data_p: pd.DataFrame, minu_cut: int) -> int:
        """Processa o tempo dos dados."""
        if 'HORA' in data_p.columns:
            return Features.settimeHor(data_p, minu_cut)
        return 0
        
    def process_date(self, data_p: pd.DataFrame) -> List[str]:
        """Processa as datas dos dados."""
        date_column = 'DATA' if 'DATA' in data_p.columns else 'DATA '
        
        if date_column in data_p.columns:
            return [
                '/'.join(
                    str(part).replace(' ', '').replace('\\', '').replace('n', '').rstrip().lstrip()
                    for part in str(date).split('-')
                )
                for date in data_p[date_column]
            ]
        return []
        
    def save_class_features(self, l_app: lacevApp, path_c: str, classe: int, resu: List) -> None:
        """Salva as features de uma classe."""
        try:
            # Verifica se o diretório existe
            feature_path = os.path.join(l_app.getGfilePathFe(), path_c)
            if not os.path.exists(feature_path):
                print(f"Diretório não encontrado: {feature_path}")
                return
                
            # Busca todos os arquivos CSV no diretório
            all_files = glob.glob(os.path.join(feature_path, "*.csv"))
            
            if not all_files:
                print(f"Nenhum arquivo CSV encontrado em: {feature_path}")
                return
                
            # Lê todos os arquivos CSV
            dataframes = []
            for filename in all_files:
                try:
                    df = pd.read_csv(filename, index_col=None, header=0)
                    if not df.empty:  # Verifica se o DataFrame não está vazio
                        dataframes.append(df)
                except Exception as e:
                    print(f"Erro ao ler arquivo {filename}: {str(e)}")
                    continue
            
            if not dataframes:  # Verifica se há DataFrames para concatenar
                print("Nenhum DataFrame válido encontrado para concatenar")
                return
                
            # Concatena os DataFrames
            dados = pd.concat(dataframes, axis=0, ignore_index=True)
            
            # Salva o resultado
            path_df = os.path.join(l_app.getGfilePathFe(), f'df{classe}.csv')
            dados.to_csv(path_df, index=False)
            print(f"Features salvas com sucesso em: {path_df}")
            
        except Exception as e:
            error_msg = f"Erro ao salvar features da classe {classe}: {str(e)}"
            print(error_msg)
            messagebox.showerror("Erro", error_msg)
            
    def action_function(self, value: str) -> None:
        """Executa a ação selecionada no menu dropdown."""
        self.root.directory = filedialog.askdirectory()
        if not self.root.directory:
            return
            
        l_app = lacevApp()
        msg = l_app.startLacevApp(self.root.directory)
        
        if msg['erro']:
            sgmsg = l_app.startgraph()
            if sgmsg['erro']:
                self.handle_plot_action(value, l_app)
            else:
                messagebox.showerror(sgmsg['msg'])
        else:
            messagebox.showerror(msg['msg'])
            
    def handle_plot_action(self, value: str, l_app: lacevApp) -> None:
        """Manipula a ação de plotagem selecionada."""
        from utius import Plots
        
        self.inst_text.set("loading...")
        self.browse_bin.destroy()
        
        plots = Plots()
        filepath_c = l_app.getfilepathC()
        
        plot_functions = {
            'FFT-PSD': plots.plotFftPsd,
            'TDAF': plots.plotTDAF,
            'TDAFLog': plots.plotTDAFLog
        }
        
        if value in plot_functions:
            fig = plot_functions[value](filepath_c, None)
            if fig:
                messagebox.showerror('Could not save graphics')
            else:
                l_app.explore(l_app.getfilePathG())
                
    def run(self) -> None:
        """Inicia a aplicação."""
        self.root.mainloop()

if __name__ == "__main__":
    app = LacevAppGUI()
    app.run()