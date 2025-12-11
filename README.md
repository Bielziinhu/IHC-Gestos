# 🖱️ Projetos de IHC

Esse repositório possui alguns projetos que usam apenas uma webcam e gestos manuais. Desenvolvido em **Python**, utiliza a biblioteca **MediaPipe** para rastreamento de mãos e **PyAutoGUI** para interação com o sistema operacional.

## ✨ Funcionalidades do Mouse

* **Controle de Cursor:** Mova o mouse apontando com o dedo indicador.
* **Clique Esquerdo & Arrastar:** Pinça com Indicador + Polegar (Suporte para segurar).
* **Clique Direito:** Dedo médio com dedo indicador.
* **Estabilidade:** Algoritmo de suavização para evitar tremores no cursor.

## ✨ Funcionalidades do Corpo

* **Movimento do Pé:** Mova-se o corpo para que a câmera possa capturar os movimentos.

## 🛠️ Pré-requisitos

Certifique-se de ter o **Python 3.10.0** instalado em sua máquina.

### Instalação das Dependências

Abra seu terminal ou prompt de comando na pasta do projeto e execute:

```bash
pip install opencv-python mediapipe pyautogui numpy
