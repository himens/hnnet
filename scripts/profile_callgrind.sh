#!/usr/bin/env bash
# Profila con valgrind/callgrind una sola regione di un binario hnnet, ignorando il setup (es. connect()).
#
# Come usarlo:
#   1. Nel sorgente da profilare, includere <valgrind/callgrind.h> e circondare la regione
#      di interesse con:
#          CALLGRIND_START_INSTRUMENTATION;
#          classifier.train(samples, rule);
#          CALLGRIND_STOP_INSTRUMENTATION;
#      (queste macro sono no-op se il binario non gira sotto valgrind: si possono lasciare nel
#      codice permanentemente, oppure aggiungerle solo temporaneamente per la sessione di profiling.)
#   2. Se il setup (es. NNet::connect()) è pesante, ridurre temporaneamente le dimensioni
#      del dataset/rete in modo che l'esecuzione sotto valgrind (10-50x più lenta) resti
#      in tempi ragionevoli: la regione instrumentata resta comunque rappresentativa.
#   3. Eseguire questo script passando il path del binario:
#          scripts/profile_callgrind.sh build/bin/hnnet-backprop-mnist
#
# Output:
#   - /tmp/callgrind.out.<pid>      : dati grezzi di profiling (eventi Ir per istruzione)
#   - /tmp/callgrind_annotate.txt   : report per funzione, ordinato per costo
#   - stdout                        : le prime N funzioni più costose (per comodità)
#
# Note:
#   - Serve compilare con `-g` (simboli di debug) per avere nomi di funzione/riga leggibili
#     invece di indirizzi/mangled names parziali.
#   - `--instr-atstart=no` disattiva la raccolta dati finché non viene chiamato
#     CALLGRIND_START_INSTRUMENTATION nel codice: così il tempo di setup (es. lettura CSV,
#     connect() O(n^2)) non inquina il profilo della regione che interessa davvero.

set -euo pipefail

BINARY="${1:?Uso: $0 <path-al-binario> [top-N=20]}"
TOP_N="${2:-20}"
OUT_DIR="${OUT_DIR:-/tmp}"
CALLGRIND_OUT="${OUT_DIR}/callgrind.out"
ANNOTATE_OUT="${OUT_DIR}/callgrind_annotate.txt"

rm -f "${CALLGRIND_OUT}"

echo ">> Eseguo ${BINARY} sotto valgrind/callgrind (instr-atstart=no)..."
valgrind \
    --tool=callgrind \
    --instr-atstart=no \
    --callgrind-out-file="${CALLGRIND_OUT}" \
    "${BINARY}"

echo ">> Genero il report annotato in ${ANNOTATE_OUT}..."
callgrind_annotate "${CALLGRIND_OUT}" > "${ANNOTATE_OUT}"

echo ">> Prime ${TOP_N} funzioni per costo (istruzioni retirate, Ir):"
echo "--------------------------------------------------------------------------------"
# Estrae solo le righe "percentuale (conteggio) simbolo" dal blocco di report principale,
# saltando l'intestazione e limitando l'output alle prime TOP_N righe utili.
awk '/^Ir /{found=1; next} found && NF' "${ANNOTATE_OUT}" | head -n "${TOP_N}"
echo "--------------------------------------------------------------------------------"
echo ">> Report completo: ${ANNOTATE_OUT}"
echo ">> Dati grezzi:      ${CALLGRIND_OUT} (ispezionabili anche con: callgrind_annotate --tree=both ${CALLGRIND_OUT})"
