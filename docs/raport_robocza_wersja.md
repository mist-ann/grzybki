# Raport z `Grzybków`

## 1. Cel projektu

Projekt implementuje algorytm **Fungal Growth Optimizer (FGO)** do optymalizacji idealizowanej sieci kolejowej.

Dany jest zbiór punktów reprezentujących miasta lub terminale. Algorytm szuka takiego układu punktów pośrednich i połączeń, aby otrzymać sieć:

- możliwie krótką,
- zapewniającą krótkie trasy między terminalami,
- odporną na awarię pojedynczych krawędzi.

Projekt jest eksperymentem z metaheurystyką. Nie jest jeszcze narzędziem do rzeczywistego projektowania linii kolejowych.

Główna dokumentacja znajduje się w `README.md` oraz `algorithm_spec.md`.

---

# 2. Ogólna architektura

Projekt składa się z kilku warstw:

```text
dane wejściowe
      |
      v
Instance + Projection
      |
      v
NetworkProblem
      |
      v
dekodowanie wektora do grafu
      |
      v
metryki C, D, FT
      |
      v
funkcja celu J
      |
      v
FGOOptimizer
      |
      v
wyniki, checkpointy, wykresy
```

Najważniejsze katalogi:

```text
src/grzybki/
├── config.py          konfiguracja algorytmu i wag
├── cli.py             interfejs wiersza poleceń
├── commands.py        wykonanie optymalizacji i wizualizacji
├── problem/
│   ├── instance.py    dane terminali
│   ├── projection.py  przeliczanie współrzędnych
│   ├── graph.py       budowa grafu
│   ├── decoder.py     dekodowanie rozwiązania
│   ├── metrics.py     obliczanie metryk
│   └── comparison.py  porównanie z liniami referencyjnymi
├── fgo/
│   ├── optimizer.py   główna pętla optymalizatora
│   ├── operators.py   operatory FGO
│   └── rng.py         generator liczb losowych
├── export/            zapis wyników
└── viz/               wykresy, mapy i animacje
```

---

# 3. Dane wejściowe

Dane opisują terminale, czyli punkty, które muszą zostać połączone.

Przykładowy terminal:

```json
{
  "name": "Warsaw",
  "lon": 21.0111,
  "lat": 52.23
}
```

Obsługiwane są dwa typy współrzędnych:

## Współrzędne geograficzne

Długość i szerokość geograficzna:

```text
lon, lat
```

Są przeliczane na lokalny układ płaski.

## Współrzędne znormalizowane

Bezpośrednio podane punkty w kwadracie:

```text
x, y ∈ [0, 1]
```

Klasa `Instance` w `instance.py` odpowiada za:

- wczytanie pliku JSON,
- walidację terminali,
- przechowywanie nazw i współrzędnych,
- projekcję współrzędnych,
- tworzenie instancji syntetycznych.

Dostępne są instancje:

- polskie miejscowości, np. `poland_15`, `poland_30`,
- punkty losowe,
- klastry,
- pierścienie,
- siatki,
- regularne wielokąty.

---

# 4. Projekcja współrzędnych

Algorytm nie działa bezpośrednio na długości i szerokości geograficznej.

Współrzędne są zamieniane na płaszczyznę przez:

- lokalną projekcję equirectangular — domyślnie,
- EPSG:2180 — opcjonalnie, przez `pyproj`.

Następnie zakres jest normalizowany do:

```text
[0, 1] × [0, 1]
```

Ważną cechą obecnej implementacji jest normalizacja izotropowa. Oznacza to, że obie osie mają tę samą skalę, więc odległość w pionie i poziomie jest porównywalna.

Klasa projekcji znajduje się w `projection.py`.

Dla danych geograficznych można także uzyskać przybliżoną długość sieci w kilometrach:

```text
C_km = C * km_per_unit
```

Jednostki kilometrowe są informacyjne — funkcja celu jest znormalizowana i nie zależy bezpośrednio od skali.

---

# 5. Reprezentacja rozwiązania

Każdy osobnik populacji jest wektorem:

```text
x = (x1, y1, x2, y2, ..., xK, yK, theta)
```

Przykład:

- 15 terminali,
- domyślnie `K = n - 2 = 13`,
- wymiar rozwiązania:

```text
2 * 13 + 1 = 27
```

Znaczenie elementów:

- pierwsze `2K` liczb określa położenie punktów Steिनera,
- ostatnia liczba `theta` określa liczbę dodatkowych krawędzi.

Punkty Steिनera są sztucznymi węzłami pośrednimi. Nie reprezentują konkretnych miast ani stacji.

---

# 6. Dekodowanie rozwiązania do grafu

Dekoder w `decoder.py` zamienia wektor na rzeczywisty graf.

## Krok 1 — połączenie terminali i punktów Steिनera

Powstaje zbiór:

```text
Q = terminale + punkty Steिनera
```

## Krok 2 — minimalne drzewo rozpinające

Na wszystkich punktach budowane jest euklidesowe MST.

MST zapewnia połączenie wszystkich punktów przy minimalnej sumie długości krawędzi.

## Krok 3 — triangulacja Delaunaya

Algorytm wyznacza potencjalne dodatkowe krawędzie na podstawie triangulacji Delaunaya.

Jeżeli triangulacja się nie powiedzie, na przykład przez degenerację punktów, używany jest graf pełny jako rozwiązanie awaryjne.

## Krok 4 — dodatkowe krawędzie

Liczba dodanych krawędzi wynosi:

```text
floor(theta * m)
```

gdzie `m` oznacza liczbę krawędzi Delaunaya, których nie ma już w MST.

Przykładowo:

- `theta = 0` — brak dodatkowych krawędzi,
- `theta = 0.5` — około połowa kandydatów,
- `theta = 1` — wszystkie dostępne krawędzie.

## Krok 5 — usunięcie niepotrzebnych punktów

Jeżeli punkt Steिनera ma stopień równy 1, jest usuwany.

Terminale nigdy nie są usuwane.

Za budowę grafu odpowiada `graph.py`.

---

# 7. Metryki jakości

Dla każdego zdekodowanego grafu liczone są trzy podstawowe wartości.

## 7.1. Koszt `C`

Koszt to suma długości wszystkich krawędzi:

```text
C = suma długości krawędzi
```

Mniejsza wartość oznacza krótszą sieć.

---

## 7.2. Objazd `D`

Dla każdej pary terminali obliczana jest wartość:

```text
d_G(i,j) / d_E(i,j)
```

gdzie:

- `d_G` — najkrótsza trasa po grafie,
- `d_E` — bezpośrednia odległość euklidesowa.

Następnie wyznaczana jest średnia dla wszystkich par terminali.

Interpretacja:

- `D = 1` — trasy są praktycznie tak krótkie jak linie proste,
- większe `D` — sieć powoduje większe objazdy.

Do obliczeń najkrótszych ścieżek używany jest `scipy.sparse.csgraph.dijkstra`.

---

## 7.3. Odporność `FT`

`FT` oznacza część krawędzi, których usunięcie nie rozdziela terminali na różne komponenty.

W uproszczeniu:

```text
FT = liczba krawędzi odpornych / liczba wszystkich krawędzi
```

Krawędź, której usunięcie rozdziela terminale, jest traktowana jak krytyczny most.

Przykładowo:

- drzewo ma zwykle niskie `FT`,
- graf z dodatkowymi obwodami ma większe `FT`.

Implementacja wykorzystuje algorytm Tarjana do wykrywania mostów.

Metryki są zaimplementowane w `metrics.py`.

---

# 8. Funkcja celu

Domyślna funkcja celu to:

```text
J = 0.4 * C / C_MST
  + 0.4 * D / D_MST
  + 0.2 * (1 - FT)
```

Znaczenie:

- pierwszy składnik karze za długą sieć,
- drugi karze za objazdy,
- trzeci karze za brak odporności.

`C_MST` i `D_MST` są liczone dla drzewa zbudowanego tylko z terminali.

Dzięki temu wynik jest względny:

```text
J = 1
```

oznacza wynik odpowiadający rozwiązaniu bazowemu.

Niższy wynik jest lepszy.

Wagi można zmieniać:

```bash
--w1 0.4 --w2 0.4 --w3 0.2
```

Klasa konfiguracji znajduje się w `config.py`.

---

# 9. Działanie FGO

Główny optymalizator znajduje się w `optimizer.py`.

## Inicjalizacja

1. Tworzona jest populacja losowych wektorów.
2. Każdy wektor ma wartości z przedziału `[0, 1]`.
3. Każdy osobnik jest oceniany.
4. Zapisywany jest najlepszy znaleziony osobnik.

Domyślna populacja ma 30 osobników.

## Kolejne iteracje

W każdej iteracji algorytm wybiera jedną z dwóch głównych faz:

- wzrost końcówki — `tip growth`,
- wzrost rozgałęzienia — `branch growth`.

### Tip growth

Może wykonać:

- eksplorację,
- eksploatację.

Eksploracja przesuwa osobnika na podstawie różnicy dwóch innych osobników:

```text
candidate = S_i + G * (S_a - S_b)
```

Eksploatacja wykorzystuje:

- najlepsze rozwiązanie globalne,
- wybranych sąsiadów,
- siłę odżywiania,
- losowe perturbacje.

### Branch growth

Może wykonać:

- `branching`,
- `germination`.

Rozgałęzianie tworzy kandydata z różnic pomiędzy osobnikami.

Kiełkowanie wykorzystuje najlepszy osobnik, innych członków populacji i losowy kierunek wzrostu.

Operatory są w `operators.py`.

## Selekcja

Nowe rozwiązanie jest przyjmowane tylko wtedy, gdy jest ściśle lepsze:

```text
score(candidate) < score(current)
```

Rozwiązania o identycznym wyniku są odrzucane.

Wszystkie wektory są przycinane do `[0, 1]`.

---

# 10. Parametry algorytmu

Najważniejsze ustawienia:

| Parametr | Domyślna wartość | Znaczenie |
|---|---:|---|
| `population` | 30 | liczba osobników |
| `max-evals` | 20 000 | maksymalna liczba ocen |
| `K` | `n - 2` | liczba punktów Steिनera |
| `M` | 0.6 | próg eksploracji |
| `Ep` | 0.7 | prawdopodobieństwo dodatkowej eksploracji |
| `R` | 0.9 | próg ruchu w stronę najlepszego |
| `w1` | 0.4 | waga kosztu |
| `w2` | 0.4 | waga objazdu |
| `w3` | 0.2 | waga odporności |
| `seed` | 42 | ziarno losowe |

Konfiguracja kontroluje także:

- częstotliwość snapshotów,
- częstotliwość checkpointów,
- sposób obliczania siły odżywiania,
- zapisywanie wektorów w historii.

---

# 11. Budżet obliczeniowy

Liczba ocen obejmuje również inicjalizację populacji.

Przykładowo dla:

```text
population = 30
max-evals = 20 000
```

wykonywanych jest dokładnie 20 000 ocen funkcji celu.

Ważne, ponieważ jedna ocena obejmuje:

- budowę MST,
- triangulację Delaunaya,
- wybór dodatkowych krawędzi,
- usuwanie punktów,
- obliczenie najkrótszych ścieżek,
- analizę mostów.

---

# 12. Reprodukowalność

Algorytm używa `numpy.random.Generator`.

Przy tych samych:

- danych wejściowych,
- parametrach,
- ziarnie losowym,
- wersjach bibliotek,

historia ocen powinna być deterministyczna.

Przykład:

```bash
grzybki optimize data/instances/poland_15.json \
  --seed 42 \
  --max-evals 20000 \
  --out runs/poland_15_seed42
```

Ziarno `-1` oznacza losowe ziarno.

---

# 13. Zapisywane wyniki

Każdy eksperyment jest zapisywany jako katalog:

```text
runs/<nazwa>/
├── manifest.json
├── summary.json
├── best.json
├── history.jsonl
├── snapshots.jsonl
├── checkpoints/
│   ├── state.npz
│   └── state.json
└── figures/
```

## `manifest.json`

Zawiera:

- dane instancji,
- parametry problemu,
- konfigurację FGO,
- wersję schematu,
- informacje o projekcji,
- środowisko uruchomieniowe.

## `summary.json`

Zawiera:

- najlepszy wynik,
- najlepszy wektor,
- liczbę ocen,
- czas działania,
- konfigurację.

## `best.json`

Zawiera szczegółowo:

- najlepszy wektor,
- wartości metryk,
- listę punktów,
- listę krawędzi,
- typy krawędzi.

## `history.jsonl`

Zawiera historię każdej oceny:

- numer oceny,
- iterację,
- fazę,
- operator,
- wynik,
- najlepszy wynik,
- informację, czy kandydat został zaakceptowany.

## Checkpointy

Pozwalają wznowić przerwaną optymalizację.

Zapisywany jest między innymi:

- stan populacji,
- wartości funkcji celu,
- najlepszy wektor,
- licznik ocen,
- stan generatora losowego.

Obsługa zapisu znajduje się w `writer.py`.

---

# 14. Interfejs użytkownika

CLI znajduje się w `cli.py`.

Dostępne są dwie główne komendy.

## Optymalizacja

```bash
grzybki optimize data/instances/poland_15.json \
  --seed 42 \
  --max-evals 20000 \
  --out runs/test
```

Można także użyć nazwy instancji:

```bash
grzybki optimize poland_15
```

## Wizualizacja

```bash
grzybki visualize runs/test \
  --out runs/test/figures
```

Generowane są:

- wykres zbieżności,
- wykres metryk,
- mapa najlepszej sieci,
- animacja,
- raport HTML.

Opcjonalnie można dodać linie referencyjne w GeoJSON:

```bash
grzybki visualize runs/test \
  --reference data/reference/osm_rail_poland.geojson
```

Linie referencyjne są tylko nakładką wizualną. Nie wpływają na funkcję celu.

---

# 15. Wizualizacja

Moduły w katalogu `viz` pozwalają przedstawić:

- terminale,
- punkty Steिनera,
- krawędzie MST,
- dodatkowe krawędzie,
- granicę Polski,
- przebieg najlepszego rozwiązania,
- zmianę wyniku w czasie.

Animacja pokazuje rozwój najlepszych rozwiązań na podstawie snapshotów.

Można również generować raport HTML zawierający wyniki i obrazy.

---

# 16. Testy

Projekt zawiera testy dla:

- metryk,
- dekodera,
- grafu,
- optymalizatora,
- operatorów FGO,
- projekcji,
- danych wejściowych,
- eksportu,
- CLI,
- wizualizacji.

Uruchomienie:

```bash
python3 -m pytest
```

Testy sprawdzają między innymi:

- poprawność obliczania MST,
- zgodność metryk z implementacją referencyjną,
- poprawność wykrywania mostów,
- zachowanie dokładnego budżetu ocen,
- deterministyczność przy ustalonym ziarnie,
- zapis i odczyt wyników,
- poprawność grafów dla szczególnych przypadków geometrycznych.

---

# 17. Wyniki wydajnościowe

Według `benchmarks.md`:

| Instancja | Oceny | Czas | Oceny/s |
|---|---:|---:|---:|
| `poland_15` | 2 000 | ok. 3,24 s | ok. 617 |
| `poland_30` | 2 000 | ok. 7,46 s | ok. 268 |
| `poland_15` | 20 000 | ok. 36,42 s | ok. 549 |
| `poland_30` | 20 000 | ok. 85,02 s | ok. 235 |

Najwięcej czasu zajmuje budowanie grafu, zwłaszcza:

- obliczanie MST,
- triangulacja Delaunaya,
- wybieranie dodatkowych krawędzi.

---

# 18. Najważniejsze ograniczenia

Zaimplementowana wersja:

1. używa geometrii euklidesowej,
2. nie zna rzeczywistych torów,
3. nie uwzględnia rzeźby terenu,
4. nie uwzględnia rzek ani obszarów chronionych,
5. używa przybliżonych centrów miejscowości,
6. nie modeluje stacji kolejowych,
7. nie uwzględnia kosztów budowy w złotówkach,
8. nie uwzględnia czasu przejazdu,
9. nie uwzględnia rozkładów jazdy i przepustowości,
10. analizuje tylko awarie pojedynczych krawędzi,
11. ma płaskie obszary funkcji celu przez `floor(theta * m)`.

Oznacza to, że wynik odpowiada na pytanie:

> Jak algorytm FGO radzi sobie z optymalizacją idealizowanej sieci punktów w przestrzeni?

Nie odpowiada jeszcze na pytanie:

> Jak zaprojektować rzeczywistą, wykonalną i opłacalną linię kolejową w Polsce?

---

# 19. Podsumowanie działania

Cały proces wygląda tak:

```text
1. Wczytaj miasta
2. Przelicz współrzędne na płaszczyznę
3. Utwórz losową populację wektorów
4. Z każdego wektora wyznacz punkty Steिनera
5. Zbuduj MST i dodatkowe krawędzie
6. Oblicz C, D i FT
7. Oblicz J
8. Wygeneruj nowe rozwiązania operatorami FGO
9. Zachowaj lepsze rozwiązania
10. Zapisz najlepszą sieć i historię
11. Wygeneruj wykresy oraz animację
```

Najważniejszą zaletą projektu jest rozdzielenie:

- **algorytmu optymalizacji**,
- **sposobu dekodowania rozwiązania**,
- **funkcji celu**,
- **wizualizacji i zapisu wyników**.

Dzięki temu można testować sam FGO niezależnie od konkretnego problemu geometrycznego, a kod jest przygotowany do późniejszego rozszerzenia — choć te rozszerzenia znajdują się jeszcze poza aktualnie zaimplementowaną wersją.