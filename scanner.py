"""
T3 - Visão Computacional (UFPR)
Scanner de documento com transformação de perspectiva.

Pipeline:
  1. Carrega a imagem (caminho passado por linha de comando).
  2. Redimensiona e pré-processa (cinza + blur).
  3. Detecta bordas (Canny) e busca o contorno do documento (4 vértices).
  4. Ordena os 4 cantos e aplica a transformação de perspectiva.

Autores:
  Gustavo Gabriel Ripka  - GRR20203935
  Arthur Barreto Godoi   - GRR20224377
"""

import sys
import os
import cv2
import numpy as np

# Tamanho máximo da imagem de trabalho (mantém o processamento rápido)
LARGURA_MAX = 800


def redimensionar(img):
    """Reduz a imagem se ela for maior que LARGURA_MAX, preservando a proporção."""
    h, w = img.shape[:2]
    if w <= LARGURA_MAX:
        return img, 1.0
    fator = LARGURA_MAX / w
    novo = cv2.resize(img, (LARGURA_MAX, int(h * fator)), interpolation=cv2.INTER_AREA)
    return novo, fator


def pre_processar(img):
    """Converte para tons de cinza e aplica blur para reduzir ruído."""
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (5, 5), 0)
    return cinza


def ordenar_cantos(pontos):
    """Ordena 4 pontos em: topo-esquerda, topo-direita, baixo-direita, baixo-esquerda."""
    pontos = pontos.reshape(4, 2).astype(np.float32)
    soma = pontos.sum(axis=1)
    dif = np.diff(pontos, axis=1)  # y - x
    te = pontos[np.argmin(soma)]    # topo-esquerda: menor x+y
    bd = pontos[np.argmax(soma)]    # baixo-direita: maior x+y
    td = pontos[np.argmin(dif)]     # topo-direita: menor y-x
    be = pontos[np.argmax(dif)]     # baixo-esquerda: maior y-x
    return np.array([te, td, bd, be], dtype=np.float32)


def encontrar_contorno_documento(img):
    """
    Encontra o contorno do documento.
    Retorna os 4 cantos ordenados, ou None se não encontrar.
    """
    cinza = pre_processar(img)

    # Detecção de bordas + dilatação para fechar pequenas falhas
    bordas = cv2.Canny(cinza, 50, 150)
    bordas = cv2.dilate(bordas, np.ones((3, 3), np.uint8), iterations=2)

    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]
    area_min = 0.1 * img.shape[0] * img.shape[1]

    for c in contornos:
        if cv2.contourArea(c) <= area_min:
            continue
        perimetro = cv2.arcLength(c, True)
        for epsilon in (0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(c, epsilon * perimetro, True)
            if len(approx) == 4:
                return ordenar_cantos(approx)
    return None


def transformar_perspectiva(img, cantos):
    """Aplica warpPerspective para obter a vista frontal do documento."""
    te, td, bd, be = cantos

    # Largura = maior distância horizontal (topo ou base)
    larg_topo = np.linalg.norm(td - te)
    larg_base = np.linalg.norm(bd - be)
    largura = int(round(max(larg_topo, larg_base)))

    # Altura = maior distância vertical (esquerda ou direita)
    alt_esq = np.linalg.norm(be - te)
    alt_dir = np.linalg.norm(bd - td)
    altura = int(round(max(alt_esq, alt_dir)))

    destino = np.array([
        [0, 0],
        [largura - 1, 0],
        [largura - 1, altura - 1],
        [0, altura - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(cantos, destino)
    corrigida = cv2.warpPerspective(img, M, (largura, altura))
    return corrigida


def main():
    if len(sys.argv) < 2:
        print("Uso: python scanner.py <caminho_da_imagem>")
        sys.exit(1)

    caminho = sys.argv[1]
    img = cv2.imread(caminho)
    if img is None:
        print(f"Erro: não foi possível ler a imagem '{caminho}'")
        sys.exit(1)

    # Diretório de saída
    saida_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saida")
    os.makedirs(saida_dir, exist_ok=True)

    # 1) Redimensiona para o tamanho de trabalho
    img, fator = redimensionar(img)

    # 2) Detecta os 4 cantos do documento
    cantos = encontrar_contorno_documento(img)
    if cantos is None:
        print("Erro: não foi possível detectar o documento na imagem.")
        sys.exit(1)
    print("Documento detectado.")

    # 3) Desenha o contorno detectado sobre a original
    contorno_img = img.copy()
    pts = cantos.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(contorno_img, [pts], True, (0, 255, 0), 3)
    for p in pts:
        cv2.circle(contorno_img, tuple(p[0]), 5, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(saida_dir, "contorno_detectado.png"), contorno_img)

    # 4) Transformação de perspectiva
    corrigida = transformar_perspectiva(img, cantos)
    cv2.imwrite(os.path.join(saida_dir, "documento_corrigido.png"), corrigida)

    print("Saídas salvas em:", saida_dir)
    print("  - contorno_detectado.png")
    print("  - documento_corrigido.png")


if __name__ == "__main__":
    main()
