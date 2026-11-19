# CarePath Lab — Roadmap didattica e tecnica

Pianificazione orientativa: **8–10 settimane** a ritmo di apprendimento; non è una scadenza. Ogni fase richiede verifiche e una breve discussione teorica prima di proseguire.

## Fase 0 — Preparazione del repository (prima sessione)
**Studiare:** virtual environment, Git, progetto Python, importazione dati, riproducibilità.  
**Fare:** verificare Python/Git, creare virtual environment, scegliere pacchetti essenziali, leggere il dataset UCI, scrivere README minimo e registro progressi.  
**Done:** l'ambiente funziona, il repo è inizializzato, il tutor ha spiegato a cosa servono tutti i file; nessun dato privato nel repo.

## Fase 1 — Data audit + SQL (settimana 1–2)
**Studiare:** missing values, qualità, distribuzioni, JOIN/CTE/aggregazioni, logica dei database relazionali.  
**Fare:** ingestione CSV, data dictionary, conteggio null/duplicati, analisi `readmitted`, PostgreSQL e query per KPI.  
**Done:** report EDA, script d'import ripetibile, query documentate e test semplici su schema e dimensioni.

## Fase 2 — Baseline scientifica (settimane 2–3)
**Stato: completata il 2026-11-12.** Report corrente: `docs/PHASE_2_REPORT.md`.

Il perimetro segue il percorso dettagliato condiviso dall'utente: include il
confronto tra regressione logistica, Random Forest e LightGBM. Il primo report
della logistica è un traguardo intermedio, non la conclusione della fase.

**Studiare:** leakage, cross-validation per paziente, regressione logistica,
bagging/Random Forest, boosting/LightGBM, class imbalance, accuracy/AUROC e curva
precision–recall, calibrazione e bootstrap per paziente. Distinguere Average
Precision (AP) dall'area PR con integrazione trapezoidale, dichiarando il metodo.

**Fare:** confronti riproducibili con coorte e partizioni comuni, feature set
espliciti e preprocessing adattato solo sui dati di training di ogni fit/fold;
metriche sulla validation, bootstrap appaiato e report comparativo. Il test finale
resta riservato fino alle scelte fissate.

**Checklist di chiusura:**

- [x] Prima pipeline logistica e riferimento dummy, split per paziente e report preliminare con sei feature.
- [x] Brier, AUROC, AP, matrice di confusione, diagnostica di calibrazione e bootstrap appaiato del Brier per logistica/dummy.
- [x] Curva precision–recall calcolata e area trapezoidale con convenzione esplicita, interpretate insieme ad AP e prevalenza; figura esportabile ancora da preparare per il report.
- [x] Cross-validation per paziente nel training per i primi confronti/configurazioni pianificati.
- [x] Set di feature più completo e motivato: 45 predittori, con esclusioni dichiarate e riferimento iniziale a sei feature riproducibile.
- [x] Random Forest implementata, spiegata e confrontata sul protocollo comune di CV.
- [x] LightGBM installato, verificato e confrontato sul protocollo comune di CV.
- [x] Esperimenti di miglioramento: tuning limitato di LightGBM e confronto con CatBoost nativo completati, incluse nuove esecuzioni senza scaling; configurazioni e risultati conservati (D-047, D-051).
- [x] Studio dei mancanti completato per il perimetro della Fase 2: imputazioni singole, gruppi di tre/sei colonne e indicatore sulla specialità, con pipeline nativa e fit interno ai fold. Candidato scelto per AP: sola medical_specialty con moda, senza indicatore (D-054). Nessuna ulteriore ricerca prima della chiusura.
- [x] Valutazione conclusiva delle procedure fissate sulla validation di sviluppo, con metriche/calibrazione, incertezza appaiata e figure PR/calibrazione esportate; test finale ancora riservato.
- [x] Report comparativo logistica/Random Forest/LightGBM/CatBoost con metriche, denominatori, incertezza e limiti: `docs/PHASE_2_REPORT.md`.
- [x] Revisione per GitHub: entry point finale distinto dai riferimenti didattici, README/report, dipendenze e 107 test, artefatti/dati/segreti/note personali esclusi; licenza MIT scelta dall'utente. Nessun commit/push/pubblicazione; ricostruzione su macchina pulita e lock transitivo non verificati.

**Done:** checklist completata, comandi riproducibili e confronto scientificamente
motivato; nessuna superiorità presunta in base alla complessità del modello.

## Fase 3 — Explainable AI e fairness (settimana 4)
**Stato: avviata il 2026-11-13.** Modello fissato dalla Fase 2; niente nuova ricerca di iperparametri.

**Studiare:** SHAP locale/globale, scala dell'output e riferimento, feature correlate e causalità; numerosità e dipendenza per paziente; recall, precision, FPR e calibrazione per sottogruppo; effetto delle soglie e limiti delle definizioni di fairness.

**Fare:** audit del CatBoost selezionato sulla validation di sviluppo, con spiegazioni e confronti per età/genere/race registrati, supporti e incertezza. Test finale riservato. MLflow è facoltativo e non un prerequisito di questa fase.

**Deliverable:** Model Audit Report riproducibile con prestazioni, spiegazioni, limitazioni e analisi della fairness, automatizzato progressivamente.

- [x] Tabella di audit con predizioni/esiti/gruppi uniti per encounter_id e coerenza paziente/esito verificata.
- [ ] Supporti per sottogruppo: ricoveri, pazienti, positivi, negativi e prevalenza; categorie ignote esplicite.
- [ ] Spiegazioni SHAP con preprocessing corretto, scala/riferimento dichiarati e limiti sulle feature correlate.
- [ ] Recall, precision, FPR e calibrazione per gruppi con denominatori e incertezza; metriche indefinite non forzate a zero.
- [x] Sensibilità a soglie comuni esplorative, senza scegliere una soglia clinica o dichiarare equità da un singolo indice.
- [ ] Report, grafici, comando riproducibile e test; nessun claim causale/clinico.

Protocollo iniziale: `docs/MODEL_AUDIT_PLAN.md`.

Primo checkpoint implementato: tabelle descrittive per genere/età/race e
calibrazione, più un intervallo bootstrap per il solo recall Female-minus-Male
a soglia .20. Comando: `python -m scripts.audit_model --output-dir artifacts/NUOVA_CARTELLA`.
L'incertezza per gli altri confronti, SHAP e il report integrato restano da completare.

## Fase 4 — Decision analysis e simulatore di capacità (settimana 5)
**Studiare:** precision@k, recall@k, capacity planning e confronto casuale; utilizzare i limiti dell'audit della Fase 3.

**Fare:** ranking e simulatore k retrospettivo, con supporti e incertezza.

**Done:** report sui limiti; nessuna stima di benefici clinici/costi evitati o pretesa causale.

## Fase 5 — Servizio e test (settimane 6–7)
**Studiare:** API REST, schema validation, pytest, logging, error handling, Docker, CI.  
**Fare:** implementare endpoint di inferenza/analisi, dashboard Streamlit, test unitari e integrazione, Docker, workflow CI.  
**Done:** un altro sviluppatore riesce ad avviare il progetto seguendo il README.

## Fase 6 — Monitoraggio e documentazione (settimana 8)
**Studiare:** data drift vs concept drift, monitoraggio, versionamento, model card, comunicazione a stakeholder.  
**Fare:** drift su dati alterati artificialmente, log e grafici, model card, demo e presentazione di 3 minuti.  
**Done:** demo funzionante, limitazioni evidenti, documentazione tecnica comprensibile.

## Fase 7 — FHIR + Agentic AI (facoltativa, settimane 9–10+)
**Studiare:** struttura base FHIR, Synthea, tool calling, restrizioni tool, valutazione delle risposte.  
**Fare:** ETL separato da pazienti sintetici e agente di analisi su metriche aggregate.  
**Done:** test automatizzati, nessuna capacità di fornire decisioni cliniche, niente uso improprio di dati sintetici per promettere generalizzazione.

## Rituale per ogni sessione
1. Leggere ROADMAP e il task aperto; se presente, usare il riepilogo locale `docs/PROGRESS.md`.
2. Discutere **un concetto teorico** importante con un esempio intuitivo.
3. Definire una modifica piccola e i criteri di accettazione.
4. Implementare o far implementare a Codex, leggere il diff, eseguire test.
5. Spiegare il codice con parole proprie e rispondere a 2 domande di verifica.
6. Aggiornare il riepilogo locale `docs/PROGRESS.md` e, per le scelte di progetto, il registro pubblico `docs/DECISIONS.md`.

## Prima sessione — task preciso
- Confermare versione Python (consigliata 3.11+), Git e ambiente macOS.
- Leggere insieme le colonne essenziali del dataset e il suo target.
- Preparare uno scheletro di repo minimale (non l'intera architettura finale).
- Scaricare/caricare un campione del CSV in `data/raw/` non tracciato da Git.
- Stampare shape, tipi, prime righe e distribuzione `readmitted`.
- Fermarsi, interpretare le prime osservazioni e decidere il task successivo.
