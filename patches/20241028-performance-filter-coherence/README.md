# Patch: Allinea filtro e pre-decimazione in modalità "Prestazioni"

## Contesto
Con l'introduzione della pre-decimazione globale del DataFrame (Issue #50) il plotting in modalità **Prestazioni** è diventato molto più rapido, ma il filtro continuava a essere eseguito direttamente sui dati decimati. Nelle preview overlay risultava quindi un tracciato filtrato diverso da quello ottenibile in **Alta fedeltà**, e in alcuni casi il recupero del risultato dalla cache generava un `KeyError` perché gli indici del DataFrame decimato non coincidevano più con quelli della serie filtrata salvata in cache.

## Problema
- `_apply_filter_cached` restituiva indiscriminatamente il valore presente in cache, anche quando la serie originale aveva indici diversi (es. `RangeIndex` completo) rispetto alla serie decimata (indici selettivi LTTB). La selezione `y_data.loc[reuse_index]` falliva dunque con `KeyError`.
- Il filtro veniva applicato alla serie già pre-decimata: il risultato perdeva accuratezza rispetto alla filtratura sui dati completi, vanificando il vantaggio funzionale della Issue #50.
- L'overlay dell'originale e le FFT mischiavano serie decimate e complete, con comportamenti diversi tra “Prestazioni” e “Alta fedeltà”.

## Soluzione
1. **Cache difensiva** – in `_apply_filter_cached` (`web_app.py:168-188`) controlliamo che l'indice della serie cache coincida con quello della serie corrente. In caso contrario la entry viene invalidata e il filtro viene ricalcolato; in questo modo evitiamo di riutilizzare risultati incompatibili quando cambia la decimazione.
2. **Fonte dati duale** – la nuova helper `_get_series_sources` (`web_app.py:1522-1530`) fornisce per ogni colonna:
   - la serie decimata da usare nel grafico (`series_plot`);
   - la serie completa (`series_full`) da usare per operazioni numeriche (filtro, FFT, overlay).
   Se il DataFrame non è stato decimato le due serie coincidono, quindi non si introduce overhead.
3. **Filtraggio coerente** – nelle tre modalità di visualizzazione (`Sovrapposto`, `Separati`, `Cascata`) applichiamo il filtro ai dati completi (`series_full`) e poi riallineiamo il risultato agli indici della decimazione tramite `.reindex(series.index)`. Il grafico conserva la leggerezza della pre-decimazione ma mostra la stessa forma d'onda ottenibile in “Alta fedeltà”. Anche FFT e overlay usano in modo esplicito la variante “full” per garantire consistenza.

## Verifiche effettuate
- Modalità **Prestazioni** con filtro Butterworth + overlay: grafico filtrato sovrapponibile all'equivalente in **Alta fedeltà** (nessuna variazione di forma).
- Cambio modalità e riesecuzione del filtro: nessun `KeyError`; la cache viene invalidata quando cambiano gli indici.
- FFT (plot sovrapposto, separati, cascata) calcolata con la serie corretta indipendentemente dal riuso cache.

## Note
- Il nuovo helper `_get_series_sources` evita doppi passaggi inutili nel caso non avvenga la pre-decimazione, mantenendo invariato il costo in modalità “Alta fedeltà”.
- L’invalidazione della cache è mirata all’entry coinvolta; le altre chiavi restano valide.
- Non sono state apportate modifiche ai preset o alle configurazioni utente.
