"""
T3 - Visão Computacional (UFPR)
Scanner de documento com transformação de perspectiva.

Pipeline:
  1. Carrega a imagem (caminho passado por linha de comando).
  2. Redimensiona e pré-processa (cinza + blur).
  3. Detecta bordas (Canny) e busca o contorno do documento (4 vértices).
  4. Ordena os 4 cantos e aplica a transformação de perspectiva.
  5. Melhora o resultado: remoção de sombra e versão P&B escaneada.

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


# ---------------------------------------------------------------------------
# 1. Pré-processamento
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Detecção do documento
# ---------------------------------------------------------------------------

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

    return _achar_quadrilatero(bordas, img)


def _achar_quadrilatero(bordas, img):
    """
    Busca, entre os 5 maiores contornos, um que aproxime bem de 4 vértices.
    O epsilon do approxPolyDP aumenta gradualmente: um valor pequeno preserva
    detalhes (mais vértices), um valor grande simplifica demais. Paramos assim
    que algum contorno grande é aproximado por exatamente 4 vértices.
    """
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


def fallback_threshold(img):
    """
    Fallback 1: o Canny pode falhar quando o documento tem sombras nas bordas
    (as bordas ficam fragmentadas). Como o papel é claro e o fundo escuro,
    usamos threshold de Otsu para separar o papel do fundo.
    """
    cinza = pre_processar(img)
    _, mascara = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Fecha pequenos buracos dentro da região do papel
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE,
                               np.ones((7, 7), np.uint8))
    return _achar_quadrilatero(mascara, img)


def fallback_minAreaRect(img):
    """
    Fallback 2 (último recurso): menor retângulo rotacionado que envolve
    o maior contorno do Canny. Funciona para documentos quase retangulares.
    """
    cinza = pre_processar(img)
    bordas = cv2.Canny(cinza, 50, 150)
    bordas = cv2.dilate(bordas, np.ones((3, 3), np.uint8), iterations=2)

    contornos, _ = cv2.findContours(bordas, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    maior = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(maior) < 0.1 * img.shape[0] * img.shape[1]:
        return None

    ret = cv2.minAreaRect(maior)
    caixa = cv2.boxPoints(ret)
    return ordenar_cantos(caixa)


# ---------------------------------------------------------------------------
# 3. Transformação de perspectiva
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4. Melhorias estilo "scan"
# ---------------------------------------------------------------------------

def remover_sombra(img):
    """
    Remove sombras dividindo a imagem pelo seu fundo estimado.
    O fundo é estimado com um closing morfológico com kernel grande.
    """
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fundo = cv2.morphologyEx(cinza, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    # Evita divisão por zero e normaliza
    sem_sombra = cv2.divide(cinza, fundo, scale=255)
    return sem_sombra


def gerar_scan_pb(img):
    """Gera a versão P&B 'escaneada' com threshold adaptativo."""
    sem_sombra = remover_sombra(img)
    scan = cv2.adaptiveThreshold(sem_sombra, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 10)
    return scan


# ---------------------------------------------------------------------------
# 5. Programa principal
# ---------------------------------------------------------------------------

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

    # 2) Detecta os 4 cantos do documento (Canny -> fallback Otsu -> fallback retângulo)
    cantos = encontrar_contorno_documento(img)
    metodo = "Canny + approxPolyDP (4 vértices)"
    if cantos is None:
        print("Aviso: Canny não encontrou 4 vértices. Tentando fallback por threshold (Otsu).")
        cantos = fallback_threshold(img)
        metodo = "fallback Otsu + approxPolyDP"
    if cantos is None:
        print("Aviso: threshold também falhou. Usando fallback minAreaRect.")
        cantos = fallback_minAreaRect(img)
        metodo = "fallback minAreaRect"
    if cantos is None:
        print("Erro: não foi possível detectar o documento na imagem.")
        sys.exit(1)
    print(f"Documento detectado ({metodo}).")

    # 3) Desenha o contorno detectado sobre a original (para o relatório)
    contorno_img = img.copy()
    pts = cantos.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(contorno_img, [pts], True, (0, 255, 0), 3)
    for p in pts:
        cv2.circle(contorno_img, tuple(p[0]), 5, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(saida_dir, "contorno_detectado.png"), contorno_img)

    # 4) Transformação de perspectiva
    corrigida = transformar_perspectiva(img, cantos)

    # 5) Versão colorida corrigida (com um leve realce de contraste)
    lab = cv2.cvtColor(corrigida, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    corrigida = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    cv2.imwrite(os.path.join(saida_dir, "documento_corrigido.png"), corrigida)

    # 6) Versão P&B escaneada (remoção de sombra + threshold adaptativo)
    scan = gerar_scan_pb(corrigida)
    cv2.imwrite(os.path.join(saida_dir, "documento_scan.png"), scan)

    print("Saídas salvas em:", saida_dir)
    print("  - contorno_detectado.png  (", contorno_img.shape[1], "x", contorno_img.shape[0], ")")
    print("  - documento_corrigido.png (", corrigida.shape[1], "x", corrigida.shape[0], ")")
    print("  - documento_scan.png      (", scan.shape[1], "x", scan.shape[0], ")")


if __name__ == "__main__":
    main()
