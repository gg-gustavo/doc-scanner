# T3 — Scanner de Documento com Transformação de Perspectiva

Trabalho da disciplina de **Visão Computacional** (UFPR).

**Autores:**
- Gustavo Gabriel Ripka — GRR20203935
- Arthur Barreto Godoi — GRR20224377

## O que faz

Aplicação simples em OpenCV que corrige a foto de um documento tirada na mão
para parecer escaneada, usando **transformação de perspectiva**:

1. Pré-processamento (redimensionamento, tons de cinza, blur)
2. Detecção de bordas com **Canny** + dilatação
3. Busca do maior contorno aproximado com 4 vértices (`cv2.approxPolyDP`)
   *(fallback: `minAreaRect` caso a aproximação com 4 vértices falhe)*
4. Ordenação dos 4 cantos e warp de perspectiva
   (`cv2.getPerspectiveTransform` + `cv2.warpPerspective`)
5. Melhoria estilo "scan": remoção de sombra (closing morfológico + divisão)
   e threshold adaptativo

## Como rodar

```bash
pip install opencv-python numpy
python scanner.py prova.jpeg
```

## Saídas (pasta `saida/`)

| Arquivo | Descrição |
|---|---|
| `contorno_detectado.png` | Original com o contorno/cantos detectados |
| `documento_corrigido.png` | Documento endireitado (vista frontal, colorido) |
| `documento_scan.png` | Documento em P&B com aparência de scan |

## Repositório

https://github.com/gg-gustavo/doc-scanner
