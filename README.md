# LacevApp - Plant Electrophysiology Processor (v4)

O **LacevApp** é uma plataforma acadêmica e de pesquisa científica para processamento e análise de sinais bioelétricos de plantas. O software automatiza o pipeline de filtragem de sinais eletrofisiológicos lentos, extrai características temporais e espectrais (TDAF), estima métricas da teoria do caos determinístico e realiza modelagem estatística longitudinal para determinar a significância dos efeitos de tratamentos nas plantas.

---

## 🚀 Funcionalidades Científicas e Técnicas

### 1. Pré-processamento e Filtragem
* **Filtro Notch Dinâmico**: Atenuação cirúrgica de frequências de linhas elétricas comerciais (50 Hz ou 60 Hz).
* **Filtro Passa-Banda Butterworth**: Configurado especificamente para a dinâmica de transporte iônico vegetal (0.005 Hz a 32 Hz).
* **Estabilização de Nyquist**: Ajuste automático da frequência de corte de acordo com a taxa de amostragem do hardware para evitar falhas de processamento em sinais lentos.

### 2. Normalização Z-Score Baseada em Baseline
* **Prevenção de Vazamento**: As estatísticas de média ($\mu$) e desvio padrão ($\sigma$) são calculadas de forma global e exclusiva sobre o período de linha de base (baseline/repouso).
* **Regularização de Tikhonov**: Inclusão de um termo de regularização ($\epsilon = 1 \times 10^{-8}$) no denominador para evitar divisões por zero em janelas de variabilidade nula.
* **Projeção Longitudinal**: Permite comparar indivíduos distintos eliminando variações estruturais físicas (como espessura de caule e umidade do solo).

### 3. Reconstrução Dinâmica de Atratores e Sistemas Caóticos
* **Estimação Dinâmica de Atraso ($\tau$)**: Computação do primeiro decaimento da função de autocorrelação (ACF) abaixo de $1/e$ para definir o lag ideal.
* **Estimação de Dimensão de Imersão ($m$)**: Algoritmo de *False Nearest Neighbors* (FNN) simplificado para desdobrar o atrator minimizando a sobreposição dimensional.
* **Entropia de Amostra (Sample Entropy)**: Implementação de algoritmos livres de viés de auto-comparação (superando a Entropia Aproximada clássica).
* **DFA (Detrended Fluctuation Analysis)**: Estudo do comportamento fractal do sinal com escalas logarítmicas de ajuste restritas dinamicamente ao intervalo estável $[4, N/4]$.
* **Expoentes de Lyapunov (Rosenstein & Eckmann)**: Determinação da taxa de divergência de trajetórias no espaço de fase para comprovar o caos determinístico.

### 4. Análise Espectral Premium
* **PSD via Multitaper**: Utilização de Sequências de Slepian (DPSS) ortogonais para reduzir o vazamento espectral em baixíssimas frequências.
* **Mapeamento de Oscilações Infra-Lentas (ISO)**: Isolamento das oscilações floemáticas vasculares na faixa de $0.005\text{ Hz a }0.1\text{ Hz}$ com interpolação espectral por zero-padding para janelas curtas.

### 5. Estatísticas OLS e Relatório
* **Modelagem Linear OLS**: Integração com o `statsmodels` para realizar regressões lineares ordinárias sobre os Z-Scores temporais, permitindo generalizações para a espécie vegetal sob tratamento.
* **Relatório HTML Interativo**: Geração de documentação detalhada integrando tabelas estatísticas, interpretações de características e gráficos dinâmicos.

---

## 📁 Estrutura de Pastas Esperada

Para que o pipeline de processamento funcione, a pasta base selecionada na interface deve conter a seguinte estrutura:

```
[Pasta do Experimento]/
└── raw/
    ├── Classe_Controle/
    │   └── Gravacao_01.csv  <- Contendo colunas "timestamp" e "value"
    └── Classe_Tratamento/
        └── Gravacao_02.csv
```

Ao processar, o software gerará automaticamente:
* `filtered/`: Os sinais purificados após a filtragem Notch e passa-banda.
* `features/`: Arquivos CSV com todas as 44 características brutas e normalizadas calculadas minuto a minuto por janela.
* `graphics/`: Plotagens longitudinais nos temas Claro e Escuro para todas as características.
* `results/`: Resumos de médias, regressões estatísticas OLS e o relatório interativo `report.html`.

---

## 🛠️ Instalação e Execução

### Pré-requisitos
* Python 3.10 ou superior instalado.

### 1. Clonar o repositório
```bash
git clone https://github.com/destritux/LacevApp.git
cd LacevApp
```

### 2. Configurar o ambiente virtual e dependências
```bash
python3 -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Rodar o Aplicativo
```bash
python3 main.py
```

### 4. Compilar um Executável Standalone
Para empacotar o software em um executável autônomo de clique único (`.exe` ou binário local):
```bash
python3 build_exe.py
```
*(O script gerará o arquivo dentro da pasta `dist/`)*.

---

## 📝 Licença
Este projeto é de uso acadêmico e científico.
Desenvolvido em parceria com o laboratório de Eletrofisiologia Vegetal.
