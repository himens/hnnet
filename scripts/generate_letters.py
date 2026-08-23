#!/usr/bin/env python3
"""
Genera 7 griglie (7x9) contenenti le lettere A..G.
Ogni griglia è larga 7 colonne e alta 9 righe.
Output:
 - salva `output/grid_<LETTER>.txt` con #/. (ascii)
"""

import os
from pathlib import Path

WIDTH = 7
HEIGHT = 9
LETTERS = {
    'A': [
        "..###..",
        ".#...#.",
        "#.....#",
        "#.....#",
        "#######",
        "#.....#",
        "#.....#",
        "#.....#",
        "#.....#",
    ],
    'B': [
        "######.",
        "#.....#",
        "#.....#",
        "######.",
        "#.....#",
        "#.....#",
        "#.....#",
        "#.....#",
        "######.",
    ],
    'C': [
        ".#####.",
        "#.....#",
        "#......",
        "#......",
        "#......",
        "#......",
        "#......",
        "#.....#",
        ".#####.",
    ],
    'D': [
        "######.",
        "#.....#",
        "#.....#",
        "#.....#",
        "#.....#",
        "#.....#",
        "#.....#",
        "#.....#",
        "######.",
    ],
    'E': [
        "#######",
        "#......",
        "#......",
        "#####..",
        "#......",
        "#......",
        "#......",
        "#......",
        "#######",
    ],
    'F': [
        "#######",
        "#......",
        "#......",
        "#####..",
        "#......",
        "#......",
        "#......",
        "#......",
        "#......",
    ],
    'G': [
        ".#####.",
        "#.....#",
        "#......",
        "#......",
        "#...###",
        "#.....#",
        "#.....#",
        "#.....#",
        ".#####.",
    ],
}

OUTPUT_DIR = Path("data/letters")
NUM_FONTS = 3
NOISY_PER_LETTER = 5
FLIP_PROB = 0.05

def validate_pattern(rows):
    if len(rows) != HEIGHT:
        raise ValueError(f"Pattern must have {HEIGHT} rows, got {len(rows)}")
    for r in rows:
        if len(r) != WIDTH:
            raise ValueError(f"Each row must have width {WIDTH}, got: '{r}' (len={len(r)})")
        for ch in r:
            if ch not in ('#', '.'):
                raise ValueError(f"Invalid character '{ch}' in pattern; allowed: '#' '.'")

def rows_to_matrix(rows):
    return [list(r) for r in rows]

def matrix_to_rows(mat):
    return [''.join(r) for r in mat]

def make_bold(rows):
    mat = rows_to_matrix(rows)
    h = len(mat)
    w = len(mat[0])
    out = [[mat[y][x] for x in range(w)] for y in range(h)]
    for y in range(h):
        for x in range(w):
            if mat[y][x] == '#':
                for dy, dx in ((0,1),(0,-1),(1,0),(-1,0)):
                    ny = y+dy
                    nx = x+dx
                    if 0 <= ny < h and 0 <= nx < w:
                        out[ny][nx] = '#'
    return matrix_to_rows(out)

def make_thin(rows):
    mat = rows_to_matrix(rows)
    h = len(mat)
    w = len(mat[0])
    out = [[mat[y][x] for x in range(w)] for y in range(h)]
    for y in range(h):
        for x in range(w):
            if mat[y][x] == '#':
                neighbors = 0
                for dy, dx in ((0,1),(0,-1),(1,0),(-1,0)):
                    ny = y+dy
                    nx = x+dx
                    if 0 <= ny < h and 0 <= nx < w and mat[ny][nx] == '#':
                        neighbors += 1
                if neighbors <= 1:
                    out[y][x] = '.'
    return matrix_to_rows(out)

def perturb(rows, flip_prob=FLIP_PROB):
    import random
    mat = rows_to_matrix(rows)
    h = len(mat)
    w = len(mat[0])
    out = [[mat[y][x] for x in range(w)] for y in range(h)]
    for y in range(h):
        for x in range(w):
            if random.random() < flip_prob:
                out[y][x] = '#' if mat[y][x] == '.' else '.'
    return matrix_to_rows(out)

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Prepare 3 font variants: base, bold, thin
    # font 1: base
    fonts = []
    base = {k: [row.replace(' ', '.').replace('*', '#') for row in v] for k, v in LETTERS.items()}
    for letter, pattern in base.items():
        validate_pattern(pattern)
    fonts.append(base)
    # font 2: bold
    bold = {k: make_bold(v) for k, v in base.items()}
    for v in bold.values():
        validate_pattern(v)
    fonts.append(bold)
    # font 3: thin
    thin = {k: make_thin(v) for k, v in base.items()}
    for v in thin.values():
        validate_pattern(v)
    fonts.append(thin)
    # Save train files
    for idx, font in enumerate(fonts, start=1):
        fname_txt = OUTPUT_DIR / f"train_{idx}.txt"
        with fname_txt.open('w', encoding='utf-8') as ft:
            for letter in sorted(font.keys()):
                rows = font[letter]
                ft.write(f"Letter {letter}\n")
                for r in rows:
                    ft.write(r + "\n")
                ft.write("\n")
    # Generate noisy samples per font
    for idx, font in enumerate(fonts, start=1):
        fname_txt = OUTPUT_DIR / f"noisy_{idx}.txt"
        with fname_txt.open('w', encoding='utf-8') as ft:
            for letter in sorted(font.keys()):
                noisy = perturb(font[letter])
                ft.write(f"Letter {letter}\n")
                for r in noisy:
                    ft.write(r + "\n")
                ft.write("\n")

    print(f"Saved letter grids in: {OUTPUT_DIR.resolve()}")

if __name__ == '__main__':
    main()