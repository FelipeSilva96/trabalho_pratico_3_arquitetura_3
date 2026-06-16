# Operações em CPU

## 1. GEMM Naive (função void gemm_naive(int N, const float *A, const float *B, float *C))

### Visão Geral
Este documento descreve a implementação da função `gemm_naive`, projetada para realizar a Multiplicação Geral de Matrizes (GEMM - General Matrix Multiply) em linguagem C. Esta é uma implementação "ingênua" (naive), focada na corretude algorítmica e legibilidade, sem otimizações de arquitetura de hardware como bloqueio de cache (tiling) ou instruções SIMD.

### Lógica Algorítmica
O algoritmo utiliza três laços de repetição aninhados para calcular o produto escalar entre as linhas da matriz de entrada $A$ e as colunas da matriz de entrada $B$. A operação matemática fundamental para determinar o valor de um elemento na matriz resultante $C$ segue a definição padrão:

$$C_{i,j} = \sum_{k=0}^{N-1} A_{i,k} \times B_{k,j}$$

Para lidar com o fato de que as matrizes são passadas como ponteiros de array unidimensional (`float *`), o algoritmo emprega a indexação *row-major* (ordem de linha principal). O índice linear de um elemento na linha `i` e coluna `j` de uma matriz de largura `N` é calculado usando a fórmula: `indice_linear = i * N + j`.

### Estrutura e Dependências
* **Assinatura da Função:** `void gemm_naive(int N, const float *A, const float *B, float *C)`
* **Dependências Externas:** Nenhuma. A função opera puramente com os tipos de dados primitivos fornecidos pelos parâmetros.
* **Complexidade Assintótica:**
  * **Tempo:** $O(N^3)$, impulsionado diretamente pelos três laços aninhados de tamanho $N$.
  * **Espaço:** $O(1)$ de memória auxiliar. A função não aloca memória dinamicamente; assume-se que o ponteiro $C$ aponta para um bloco de memória pré-alocado e de tamanho suficiente para armazenar $N \times N$ elementos do tipo `float`.

---

## 2. GEMM com Matriz Transposta (função void gemm_transposed(int N, const float *A, const float *BT, float *C))

### Visão Geral
Este documento descreve a implementação da função `gemm_transposed`, que realiza a Multiplicação Geral de Matrizes utilizando a versão transposta da segunda matriz de entrada (referida aqui como $BT$). Esta abordagem resolve o problema primário de ineficiência de cache da implementação ingênua, garantindo que a memória de ambas as matrizes operadas seja acessada sequencialmente no laço de repetição mais interno.

### Lógica Algorítmica
Na multiplicação de matrizes padrão, o cálculo de $C_{i,j}$ exige o produto escalar da linha $i$ de $A$ pela coluna $j$ de $B$. Quando as matrizes são linearizadas em ordem *row-major*, varrer a coluna de $B$ implica em saltos de memória do tamanho $N$, o que causa constantes falhas de cache (*cache misses*).

Ao receber a matriz $B$ transposta ($BT$), as colunas de $B$ tornam-se as linhas de $BT$. Matematicamente, a propriedade $B_{k,j} = BT_{j,k}$ é aplicada. A fórmula do cálculo se ajusta para:

$$C_{i,j} = \sum_{k=0}^{N-1} A_{i,k} \times BT_{j,k}$$

A nível de código, o acesso `B[k * N + j]` é substituído por `BT[j * N + k]`. Como o iterador $k$ varia sequencialmente no laço mais interno, tanto `A[i * N + k]` quanto `BT[j * N + k]` realizam acessos contíguos de memória, maximizando a localidade espacial e o uso eficiente da hierarquia de cache da CPU.

### Estrutura e Dependências
* **Assinatura da Função:** `void gemm_transposed(int N, const float *A, const float *BT, float *C)`
* **Dependências Externas:** Nenhuma.
* **Complexidade Assintótica:**
  * **Tempo:** $O(N^3)$. O número de operações aritméticas de ponto flutuante permanece o mesmo da versão ingênua, embora o tempo de execução prático seja significativamente menor devido à otimização de cache.
  * **Espaço:** $O(1)$ de memória auxiliar. Supõe-se que a matriz transposta $BT$ já tenha sido alocada e populada externamente à função.

---

## 3. GEMM em Blocos/Tiling (função gemm_blocked(int N, int BS, const float *A, const float *B, float *C))

### Visão Geral
Este documento detalha a implementação da função `gemm_blocked`, que aplica a técnica de *Loop Tiling* (ou particionamento em blocos) para a Multiplicação Geral de Matrizes. O objetivo fundamental desta técnica não é alterar a quantidade de operações matemáticas, mas reordenar o acesso aos dados para otimizar a interação com a hierarquia de memória da arquitetura do processador, minimizando a latência causada por sucessivos *cache misses*.

### Lógica Algorítmica
O algoritmo subdivide as matrizes inteiras de tamanho $N \times N$ em sub-matrizes menores de tamanho $BS \times BS$. O parâmetro `BS` (Block Size) deve ser ajustado com base na capacidade da memória cache (L1 ou L2) do processador alvo. O tamanho ideal permite que três blocos (um de $A$, um de $B$ e um de $C$) caibam simultaneamente no cache de dados mais rápido.

A operação é regida por seis laços de repetição aninhados. Os três laços externos (`i0`, `j0`, `k0`) navegam pelas coordenadas de origem de cada bloco. Os três laços internos (`i`, `j`, `k`) executam a multiplicação escalar estritamente dentro dos limites físicos definidos pelo bloco atual:

$$C_{i,j} = C_{i,j} + \sum_{k=k_0}^{\min(k_0+BS, N)-1} A_{i,k} \times B_{k,j}$$

A matriz $C$ precisa ser inicializada com zeros previamente à execução dos blocos, pois a natureza do algoritmo exige que os produtos parciais de diferentes blocos de $A$ e $B$ sejam acumulados em uma mesma coordenada $C_{i,j}$ ao longo de múltiplas iterações espaciais.

### Estrutura e Dependências
* **Assinatura da Função:** `void gemm_blocked(int N, int BS, const float *A, const float *B, float *C)`
* **Dependências Externas:** Nenhuma.
* **Tratamento de Limites:** O código implementa o cálculo `i_max`, `j_max` e `k_max` internamente para garantir que a função seja robusta e não acesse áreas de memória inválidas caso $N$ não seja um múltiplo exato de $BS$.
* **Complexidade Assintótica:**
  * **Tempo:** $O(N^3)$. O número de multiplicações e adições é idêntico ao da versão ingênua, mas o rendimento (*throughput*) prático é ordens de grandeza maior devido à localidade temporal maximizada.
  * **Espaço:** $O(1)$.
---

## 4. GEMM com OpenMP

### Visão Geral
Este documento descreve a implementação da função `gemm_openmp`, que utiliza a API OpenMP (Open Multi-Processing) para paralelizar o algoritmo da Multiplicação Geral de Matrizes. O objetivo é distribuir a carga de trabalho matemático entre os múltiplos núcleos físicos e lógicos do processador, reduzindo drasticamente o tempo de execução global sem alterar a complexidade fundamental do algoritmo.

### Lógica Algorítmica
A paralelização é aplicada no laço de repetição mais externo, que controla o iterador da linha `i` da matriz $A$ e da matriz resultante $C$. A diretiva `#pragma omp parallel for` instrui o compilador a gerar código que divide as $N$ iterações deste laço em blocos ("chunks") gerenciados por um conjunto de threads de execução (pool de threads).

A operação atômica iterada permanece a mesma:
$$C_{i,j} = \sum_{k=0}^{N-1} A_{i,k} \times B_{k,j}$$

#### Escopo de Variáveis em Paralelo
* **Compartilhadas (Shared):** Os ponteiros de memória $A$, $B$ e $C$, assim como a dimensão $N$, são lidos de forma simultânea. Não há condição de corrida (*race condition*) para a escrita em $C$, pois cada iteração do laço externo garante que cada thread escreva em linhas distintas da matriz de saída.
* **Privadas (Private):** Os iteradores `i`, `j`, `k` e o acumulador `sum` são privados por estarem alocados no contexto léxico dentro da diretiva paralela, garantindo que cada thread mantenha seu próprio estado durante o cálculo do produto escalar.

### Estrutura e Dependências
* **Assinatura da Função:** `void gemm_openmp(int N, const float *A, const float *B, float *C)`
* **Dependências Externas:** Requer suporte ao padrão OpenMP pelo compilador (e.g., compilar com a flag `-fopenmp` no GCC/Clang ou `/openmp` no MSVC). Dependendo da configuração, pode ser necessário incluir o cabeçalho `<omp.h>` no arquivo fonte, especialmente se funções de biblioteca específicas do OpenMP forem adicionadas posteriormente.
* **Complexidade Assintótica:**
    * **Tempo:** $O(N^3)$ em total de operações. No entanto, o tempo de parede (*wall-clock time*) decresce idealmente para $\approx O(N^3 / T)$, onde $T$ é o número de threads operando em paralelo (limitado pela Lei de Amdahl e por gargalos no barramento de memória).
    * **Espaço:** $O(1)$ de memória extra auxiliar por thread para as variáveis locais.
---

## 5. GEMM em Blocos com OpenMP

### Visão Geral
Este documento cobre a implementação da função `gemm_blocked_openmp`, que funde as duas otimizações arquiteturais anteriores: o particionamento em blocos (Tiling) para maximização da eficiência do cache L1/L2 e a diretiva OpenMP para paralelismo ao nível de thread. Esta é a abordagem padrão para obter alto rendimento (*throughput*) no cálculo matricial em CPUs multi-core sem recorrer a intrínsecos de vetorização SIMD.

### Lógica Algorítmica
O algoritmo opera dividindo a matriz em uma grade tridimensional de blocos ditada pelos iterações `i0`, `j0` e `k0`. Para extrair o paralelismo, os laços independentes devem ser distribuídos. 

A diretiva `#pragma omp parallel for collapse(2)` é utilizada sobre os dois laços mais externos (`i0` e `j0`). 
* O `collapse(2)` instrui o compilador a planificar esses dois laços aninhados em um único espaço de iteração linear de tamanho $\approx (N/BS)^2$. 
* Essa técnica previne a ociosidade das threads (subutilização dos núcleos) que ocorreria se apenas o laço `i0` fosse paralelizado em matrizes onde a dimensão $N$ não é suficientemente maior que o tamanho do bloco $BS$.
* Condição de Corrida (*Race Condition*): Evitada inerentemente pela lógica da blocagem. Cada thread passa a ser responsável pelo cálculo completo de uma sub-matriz delimitada de $C$, acumulando sequencialmente os produtos ao longo do eixo $k0$. Nenhuma área de memória de $C$ é atualizada por duas threads simultaneamente.

A inicialização do vetor linearizado de $C$ com zeros também recebe a diretiva `#pragma omp parallel for` devido à vantagem direta do processamento vetorial sem overhead significativo, preparando as posições de memória antes da carga de cálculo pesada.

### Estrutura e Dependências
* **Assinatura da Função:** `void gemm_blocked_openmp(int N, int BS, const float *A, const float *B, float *C)`
* **Dependências Externas:** Exige que a compilação ative o suporte ao OpenMP (flag `-fopenmp` no GCC/Clang).
* **Complexidade Assintótica:**
  * **Tempo:** O total matemático global se mantém em $O(N^3)$. O tempo de execução da perspectiva do usuário despenca em proporção direta a $\approx O(N^3 / T)$, otimizado pela minimização da latência de memória garantida pela blocagem.
  * **Espaço:** $O(1)$ de memória auxiliar.

# Operações em GPU

## 1. GEMM CUDA Tiled (função '__global__ void gemm_cuda_tiled(int N, const float *A, const float *B, float *C)')

### Visão Geral
Este documento descreve a implementação do kernel CUDA `gemm_cuda_tiled`, projetado para realizar a Multiplicação Geral de Matrizes (GEMM) otimizada. Ao contrário da versão ingênua na GPU, esta implementação utiliza a técnica de *tiling* (blocagem) em conjunto com a Memória Compartilhada para reduzir drasticamente a quantidade de acessos lentos à Memória Global, reaproveitando os dados previamente carregados entre as *threads* de um mesmo bloco computacional.

### Lógica Algorítmica e Paralelismo
O algoritmo divide o cálculo da matriz em "fases" ou "tiles" de tamanho fixo (`TILE x TILE`), comumente definido como $16 \times 16$:
* **Memória Compartilhada:** Duas matrizes locais (`s_A` e `s_B`) são alocadas na memória `__shared__`. Cada *thread* de um bloco carrega colaborativamente um elemento da matriz de entrada $A$ e um da matriz de entrada $B$ da memória global para a memória compartilhada.
* **Sincronização (`__syncthreads()`):** O uso da barreira de sincronização é estritamente necessário em dois momentos críticos por fase:
  1. Após o carregamento dos dados, para garantir que o *tile* inteiro esteja disponível na memória compartilhada antes de iniciar as contas matemáticas.
  2. Após a computação daquele bloco, para evitar que *threads* mais rápidas iniciem a próxima fase e sobrescrevam a memória compartilhada (condição de corrida) enquanto outras ainda estão realizando multiplicações do *tile* anterior.
* **Tratamento de Limites (Boundary Check):** Como a dimensão $N$ pode não ser perfeitamente divisível pelo tamanho predefinido do bloco, as *threads* cujos índices excedem os limites válidos da matriz injetam o valor de segurança `0.0f` na memória compartilhada (padding virtual), garantindo que lixo de memória não corrompa o cálculo final.

### Estrutura e Dependências

* **Assinatura da Função:** `__global__ void gemm_cuda_tiled(int N, const float *A, const float *B, float *C)`
* **Localidade de Memória (@param):** * Os ponteiros `A`, `B` e `C` devem obrigatoriamente estar alocados na **Memória do Device** (VRAM). Tentar desreferenciar um ponteiro do *Host* gerará falha de segmentação no acelerador.
  
* **Configuração de Lançamento (Launch Config):**
  * O *kernel* foi projetado exigindo uma topologia de grade (Grid) bidimensional (2D) e blocos bidimensionais (2D).
  * @note Requisito de lançamento sugerido: `dim3 block(TILE, TILE)` (onde o valor de TILE precisa corresponder à macro definida na compilação, tipicamente 16).
  * Requisito da Grade: `dim3 grid((N + TILE - 1)/TILE, (N + TILE - 1)/TILE)`.

* **Gargalos e Análise de Desempenho:**
  * O padrão de acesso à Memória Global é otimizado e coalescido durante a fase de transferência paralela para a Memória Compartilhada.
  * Em comparação à versão de CUDA *Naive*, o número total de requisições de leitura diretas à Memória Global é reduzido por um fator equivalente à variável `TILE`, mitigando o maior limitador de desempenho deste algoritmo computacional, conhecido na literatura como *Memory Wall* (Gargalo de Memória).
---
## 2. GEMM CUDA Tiled (função `__global__ void gemm_cuda_tiled(int N, const float *A, const float *B, float *C)`)

### Visão Geral
Este documento descreve a implementação do kernel CUDA `gemm_cuda_tiled`, projetado para realizar a Multiplicação Geral de Matrizes (GEMM) otimizada. Ao contrário da versão ingênua na GPU, esta implementação utiliza a técnica de *tiling* (blocagem) em conjunto com a Memória Compartilhada para reduzir drasticamente a quantidade de acessos lentos à Memória Global, reaproveitando os dados previamente carregados entre as *threads* de um mesmo bloco computacional.

### Lógica Algorítmica e Paralelismo
O algoritmo divide o cálculo da matriz em "fases" ou "tiles" de tamanho fixo (`TILE x TILE`), comumente definido como 16x16:
* **Memória Compartilhada:** Duas matrizes locais (`s_A` e `s_B`) são alocadas na memória `__shared__`. Cada *thread* de um bloco carrega colaborativamente um elemento da matriz de entrada $A$ e um da matriz de entrada $B$ da memória global para a memória compartilhada.
* **Sincronização (`__syncthreads()`):** O uso da barreira de sincronização é estritamente necessário em dois momentos críticos por fase:
  1. Após o carregamento dos dados, para garantir que o *tile* inteiro esteja disponível na memória compartilhada antes de iniciar as contas matemáticas.
  2. Após a computação daquele bloco, para evitar que *threads* mais rápidas iniciem a próxima fase e sobrescrevam a memória compartilhada (condição de corrida) enquanto outras ainda estão realizando multiplicações do *tile* anterior.
* **Tratamento de Limites (Boundary Check):** Como a dimensão $N$ pode não ser perfeitamente divisível pelo tamanho predefinido do bloco, as *threads* cujos índices excedem os limites válidos da matriz injetam o valor de segurança `0.0f` na memória compartilhada (padding virtual), garantindo que lixo de memória não corrompa o cálculo final.

### Estrutura e Dependências

* **Assinatura da Função:** `__global__ void gemm_cuda_tiled(int N, const float *A, const float *B, float *C)`
* **Localidade de Memória (@param):** Os ponteiros `A`, `B` e `C` devem obrigatoriamente estar alocados na **Memória do Device** (VRAM). Tentar desreferenciar um ponteiro do *Host* gerará falha de segmentação no acelerador.
  
* **Configuração de Lançamento (Launch Config):**
  * O *kernel* foi projetado exigindo uma topologia de grade (Grid) bidimensional (2D) e blocos bidimensionais (2D).
  * Requisito de lançamento sugerido: `dim3 block(TILE, TILE)` (onde o valor de TILE precisa corresponder à macro definida na compilação, tipicamente 16).
  * Requisito da Grade: `dim3 grid((N + TILE - 1)/TILE, (N + TILE - 1)/TILE)`.

* **Gargalos e Análise de Desempenho:**
  * O padrão de acesso à Memória Global é otimizado e coalescido durante a fase de transferência paralela para a Memória Compartilhada.
  * Em comparação à versão de CUDA *Naive*, o número total de requisições de leitura diretas à Memória Global é reduzido por um fator equivalente à constante `TILE`, mitigando o maior limitador de desempenho deste algoritmo computacional, conhecido na literatura como *Memory Wall* (Gargalo de Memória).