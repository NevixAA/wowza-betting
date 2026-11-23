# P11 v3 price coverage before / after (code PR, P11 v2 prediction rows)

Owner 2026-11-23: HT prices = "Wire in + re-measure (Recommended)"; fallback books = "Extend to all markets (Recommended)" (owner 2026-11-23). Counts only: no prices, keys or club names. No credits spent. Serving bars and tier config unchanged.

- Universe: the P11 v2 prediction universes (the rows TP-2026-11-18-1ddfc9f1 was fitted on), so before vs after differs only by the price rules
- Store: wowza-exec `odds-history` @ `2ea84459`.
- Before = P11 v2 price rules: BET tiers BT, BTU, OUU; fallback books on BTTS only.
- After = v3 rules: BET tiers BT, BTU, OUU, HTB; O1 fallback (William Hill > 1xBet > Ladbrokes after Bet365 > Pinnacle) on BTTS, O1.5, O3.5, HT 0.5/1.5 (both tracks) + New-Format O/U 2.5; 00:00:00-stamped history pairs dropped.

- standard: price source before {'oddsapi_hist': 33223, 'own': 26677}, after {'oddsapi_hist': 40159, 'own': 26677}; join integrity failures after 0 of 40159 history rows; 00:00-stamped history pairs dropped 0.
- new_format: price source before {'oddsapi_hist': 24465, 'own': 3061}, after {'oddsapi_hist': 31598, 'own': 3061}; join integrity failures after 0 of 31598 history rows; 00:00-stamped history pairs dropped 0.

Rows = fit (< 2026-09-01) + validation (2026-09-01..11-08) rows with a result. EFL first.

| Track | League | Market | Rows | Priced before (v2 rules) | Priced after (v3 rules) | Added | After: Pinnacle / WH / 1xBet / Ladbrokes (history rows) |
|---|---|---|---|---|---|---|---|
| standard | Championship | btts | 2,146 | 1,590 | 1,590 | +0 | 0 / 1109 / 395 / 8 |
| standard | Championship | ht_over05 | 1,221 | 65 | 477 | +412 | 412 / 0 / 0 / 0 |
| standard | Championship | ht_over15 | 1,221 | 64 | 493 | +429 | 429 / 0 / 0 / 0 |
| standard | Championship | over15 | 2,146 | 1,077 | 1,538 | +461 | 995 / 269 / 0 / 192 |
| standard | Championship | over25 | 2,146 | 2,133 | 2,133 | +0 | 7 / 0 / 0 / 0 |
| standard | Championship | over35 | 2,146 | 1,082 | 1,449 | +367 | 987 / 0 / 0 / 367 |
| standard | League One | btts | 2,157 | 1,409 | 1,409 | +0 | 0 / 1062 / 274 / 0 |
| standard | League One | ht_over05 | 1,196 | 59 | 458 | +399 | 399 / 0 / 0 / 0 |
| standard | League One | ht_over15 | 1,196 | 55 | 509 | +454 | 454 / 0 / 0 / 0 |
| standard | League One | over15 | 2,157 | 832 | 1,224 | +392 | 744 / 342 / 0 / 50 |
| standard | League One | over25 | 2,157 | 2,146 | 2,146 | +0 | 9 / 0 / 0 / 0 |
| standard | League One | over35 | 2,157 | 833 | 1,041 | +208 | 741 / 0 / 0 / 208 |
| standard | League Two | btts | 2,165 | 1,527 | 1,527 | +0 | 0 / 1143 / 304 / 0 |
| standard | League Two | ht_over05 | 1,194 | 59 | 576 | +517 | 517 / 0 / 0 / 0 |
| standard | League Two | ht_over15 | 1,194 | 56 | 614 | +558 | 558 / 0 / 0 / 0 |
| standard | League Two | over15 | 2,165 | 960 | 1,347 | +387 | 872 / 319 / 0 / 68 |
| standard | League Two | over25 | 2,165 | 2,155 | 2,155 | +0 | 10 / 0 / 0 / 0 |
| standard | League Two | over35 | 2,165 | 963 | 1,164 | +201 | 868 / 0 / 0 / 201 |
| standard | National League | btts | 2,176 | 70 | 70 | +0 | 0 / 0 / 0 / 0 |
| standard | National League | ht_over05 | 1,189 | 42 | 42 | +0 | 0 / 0 / 0 / 0 |
| standard | National League | ht_over15 | 1,189 | 43 | 43 | +0 | 0 / 0 / 0 / 0 |
| standard | National League | over15 | 2,176 | 59 | 59 | +0 | 0 / 0 / 0 / 0 |
| standard | National League | over25 | 2,176 | 2,144 | 2,144 | +0 | 0 / 0 / 0 / 0 |
| standard | National League | over35 | 2,176 | 69 | 69 | +0 | 0 / 0 / 0 / 0 |
| new_format | Argentina Primera Division | btts | 2,642 | 1,064 | 1,064 | +0 | 0 / 0 / 919 / 42 |
| new_format | Argentina Primera Division | ht_over05 | 2,626 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Argentina Primera Division | ht_over15 | 2,626 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Argentina Primera Division | over15 | 2,642 | 642 | 970 | +328 | 532 / 0 / 0 / 328 |
| new_format | Argentina Primera Division | over25 | 2,642 | 644 | 967 | +323 | 520 / 0 / 0 / 323 |
| new_format | Argentina Primera Division | over35 | 2,642 | 607 | 962 | +355 | 502 / 0 / 0 / 355 |
| new_format | Austrian Bundesliga | btts | 1,038 | 571 | 571 | +0 | 0 / 397 / 151 / 0 |
| new_format | Austrian Bundesliga | ht_over05 | 1,032 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Austrian Bundesliga | ht_over15 | 1,032 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Austrian Bundesliga | over15 | 1,038 | 449 | 492 | +43 | 420 / 43 / 0 / 0 |
| new_format | Austrian Bundesliga | over25 | 1,038 | 450 | 492 | +42 | 416 / 42 / 0 / 0 |
| new_format | Austrian Bundesliga | over35 | 1,038 | 454 | 454 | +0 | 420 / 0 / 0 / 0 |
| new_format | Brazil Serie A | btts | 2,115 | 1,113 | 1,113 | +0 | 0 / 705 / 290 / 40 |
| new_format | Brazil Serie A | ht_over05 | 2,114 | 0 | 370 | +370 | 370 / 0 / 0 / 0 |
| new_format | Brazil Serie A | ht_over15 | 2,114 | 0 | 278 | +278 | 278 / 0 / 0 / 0 |
| new_format | Brazil Serie A | over15 | 2,115 | 588 | 1,071 | +483 | 515 / 257 / 0 / 226 |
| new_format | Brazil Serie A | over25 | 2,115 | 562 | 1,069 | +507 | 480 / 282 / 0 / 225 |
| new_format | Brazil Serie A | over35 | 2,115 | 586 | 956 | +370 | 515 / 0 / 0 / 370 |
| new_format | China Super League | btts | 1,372 | 333 | 333 | +0 | 0 / 94 / 213 / 0 |
| new_format | China Super League | ht_over05 | 1,291 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | China Super League | ht_over15 | 1,291 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | China Super League | over15 | 1,372 | 296 | 299 | +3 | 263 / 3 / 0 / 0 |
| new_format | China Super League | over25 | 1,372 | 309 | 309 | +0 | 277 / 0 / 0 / 0 |
| new_format | China Super League | over35 | 1,372 | 316 | 316 | +0 | 279 / 0 / 0 / 0 |
| new_format | Denmark Superliga | btts | 1,038 | 535 | 535 | +0 | 0 / 360 / 145 / 0 |
| new_format | Denmark Superliga | ht_over05 | 1,031 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Denmark Superliga | ht_over15 | 1,031 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Denmark Superliga | over15 | 1,038 | 386 | 441 | +55 | 357 / 53 / 0 / 2 |
| new_format | Denmark Superliga | over25 | 1,038 | 371 | 442 | +71 | 340 / 69 / 0 / 2 |
| new_format | Denmark Superliga | over35 | 1,038 | 387 | 406 | +19 | 362 / 0 / 0 / 19 |
| new_format | Finland Veikkausliiga | btts | 951 | 362 | 362 | +0 | 0 / 198 / 134 / 0 |
| new_format | Finland Veikkausliiga | ht_over05 | 945 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Finland Veikkausliiga | ht_over15 | 945 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Finland Veikkausliiga | over15 | 951 | 295 | 335 | +40 | 260 / 39 / 0 / 1 |
| new_format | Finland Veikkausliiga | over25 | 951 | 276 | 341 | +65 | 236 / 64 / 0 / 1 |
| new_format | Finland Veikkausliiga | over35 | 951 | 300 | 316 | +16 | 262 / 0 / 0 / 16 |
| new_format | Ireland Premier Division | btts | 980 | 295 | 295 | +0 | 0 / 101 / 159 / 0 |
| new_format | Ireland Premier Division | ht_over05 | 978 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Ireland Premier Division | ht_over15 | 978 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Ireland Premier Division | over15 | 980 | 261 | 270 | +9 | 224 / 9 / 0 / 0 |
| new_format | Ireland Premier Division | over25 | 980 | 255 | 268 | +13 | 217 / 13 / 0 / 0 |
| new_format | Ireland Premier Division | over35 | 980 | 261 | 263 | +2 | 222 / 0 / 0 / 2 |
| new_format | Japan J-League | btts | 1,870 | 620 | 620 | +0 | 0 / 431 / 186 / 0 |
| new_format | Japan J-League | ht_over05 | 1,858 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Japan J-League | ht_over15 | 1,858 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Japan J-League | over15 | 1,870 | 550 | 593 | +43 | 543 / 42 / 0 / 1 |
| new_format | Japan J-League | over25 | 1,870 | 512 | 594 | +82 | 502 / 81 / 0 / 1 |
| new_format | Japan J-League | over35 | 1,870 | 550 | 564 | +14 | 543 / 0 / 0 / 14 |
| new_format | K-League 1 | btts | 1,188 | 382 | 382 | +0 | 0 / 322 / 51 / 0 |
| new_format | K-League 1 | ht_over05 | 1,188 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | K-League 1 | ht_over15 | 1,188 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | K-League 1 | over15 | 1,188 | 337 | 370 | +33 | 325 / 33 / 0 / 0 |
| new_format | K-League 1 | over25 | 1,188 | 314 | 368 | +54 | 300 / 54 / 0 / 0 |
| new_format | K-League 1 | over35 | 1,188 | 340 | 340 | +0 | 326 / 0 / 0 / 0 |
| new_format | Mexico Liga MX | btts | 1,830 | 597 | 597 | +0 | 0 / 420 / 111 / 3 |
| new_format | Mexico Liga MX | ht_over05 | 1,823 | 0 | 95 | +95 | 4 / 0 / 91 / 0 |
| new_format | Mexico Liga MX | ht_over15 | 1,823 | 0 | 96 | +96 | 15 / 0 / 81 / 0 |
| new_format | Mexico Liga MX | over15 | 1,830 | 447 | 657 | +210 | 370 / 137 / 71 / 2 |
| new_format | Mexico Liga MX | over25 | 1,830 | 453 | 663 | +210 | 368 / 137 / 71 / 2 |
| new_format | Mexico Liga MX | over35 | 1,830 | 451 | 604 | +153 | 372 / 0 / 145 / 8 |
| new_format | Norway Eliteserien | btts | 1,303 | 606 | 606 | +0 | 0 / 428 / 139 / 1 |
| new_format | Norway Eliteserien | ht_over05 | 1,295 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Norway Eliteserien | ht_over15 | 1,295 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Norway Eliteserien | over15 | 1,303 | 493 | 564 | +71 | 453 / 71 / 0 / 0 |
| new_format | Norway Eliteserien | over25 | 1,303 | 498 | 564 | +66 | 454 / 66 / 0 / 0 |
| new_format | Norway Eliteserien | over35 | 1,303 | 505 | 530 | +25 | 463 / 0 / 0 / 25 |
| new_format | Romanian Superliga | btts | 1,698 | 44 | 44 | +0 | 0 / 0 / 0 / 0 |
| new_format | Romanian Superliga | ht_over05 | 1,689 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Romanian Superliga | ht_over15 | 1,689 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Romanian Superliga | over15 | 1,698 | 53 | 53 | +0 | 0 / 0 / 0 / 0 |
| new_format | Romanian Superliga | over25 | 1,698 | 53 | 53 | +0 | 0 / 0 / 0 / 0 |
| new_format | Romanian Superliga | over35 | 1,698 | 50 | 50 | +0 | 0 / 0 / 0 / 0 |
| new_format | Saudi Pro League | btts | 1,420 | 46 | 46 | +0 | 0 / 0 / 27 / 0 |
| new_format | Saudi Pro League | ht_over05 | 1,420 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Saudi Pro League | ht_over15 | 1,420 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Saudi Pro League | over15 | 1,420 | 57 | 57 | +0 | 24 / 0 / 0 / 0 |
| new_format | Saudi Pro League | over25 | 1,420 | 59 | 59 | +0 | 26 / 0 / 0 / 0 |
| new_format | Saudi Pro League | over35 | 1,420 | 60 | 60 | +0 | 26 / 0 / 0 / 0 |
| new_format | Sweden Allsvenskan | btts | 1,335 | 610 | 610 | +0 | 0 / 382 / 183 / 0 |
| new_format | Sweden Allsvenskan | ht_over05 | 1,327 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Sweden Allsvenskan | ht_over15 | 1,327 | 0 | 0 | +0 | 0 / 0 / 0 / 0 |
| new_format | Sweden Allsvenskan | over15 | 1,335 | 418 | 515 | +97 | 369 / 92 / 0 / 5 |
| new_format | Sweden Allsvenskan | over25 | 1,335 | 403 | 520 | +117 | 350 / 112 / 0 / 5 |
| new_format | Sweden Allsvenskan | over35 | 1,335 | 420 | 456 | +36 | 373 / 0 / 0 / 36 |
| new_format | USA MLS | btts | 2,859 | 1,560 | 1,560 | +0 | 0 / 731 / 689 / 3 |
| new_format | USA MLS | ht_over05 | 2,844 | 0 | 390 | +390 | 141 / 0 / 249 / 0 |
| new_format | USA MLS | ht_over15 | 2,844 | 0 | 834 | +834 | 717 / 0 / 117 / 0 |
| new_format | USA MLS | over15 | 2,859 | 1,027 | 1,419 | +392 | 878 / 212 / 4 / 176 |
| new_format | USA MLS | over25 | 2,859 | 1,000 | 1,418 | +418 | 850 / 236 / 4 / 178 |
| new_format | USA MLS | over35 | 2,859 | 1,043 | 1,348 | +305 | 876 / 0 / 86 / 219 |
| standard | Belgian First Division A | btts | 1,221 | 872 | 872 | +0 | 0 / 670 / 162 / 5 |
| standard | Belgian First Division A | ht_over05 | 680 | 17 | 17 | +0 | 0 / 0 / 0 / 0 |
| standard | Belgian First Division A | ht_over15 | 680 | 23 | 23 | +0 | 0 / 0 / 0 / 0 |
| standard | Belgian First Division A | over15 | 1,221 | 818 | 870 | +52 | 783 / 26 / 0 / 26 |
| standard | Belgian First Division A | over25 | 1,221 | 1,180 | 1,180 | +0 | 0 / 0 / 0 / 0 |
| standard | Belgian First Division A | over35 | 1,221 | 821 | 864 | +43 | 783 / 0 / 0 / 43 |
| standard | Bundesliga 2 | btts | 1,166 | 942 | 942 | +0 | 0 / 830 / 82 / 0 |
| standard | Bundesliga 2 | ht_over05 | 667 | 17 | 17 | +0 | 0 / 0 / 0 / 0 |
| standard | Bundesliga 2 | ht_over15 | 667 | 26 | 26 | +0 | 0 / 0 / 0 / 0 |
| standard | Bundesliga 2 | over15 | 1,166 | 864 | 940 | +76 | 817 / 51 / 0 / 25 |
| standard | Bundesliga 2 | over25 | 1,166 | 1,161 | 1,161 | +0 | 0 / 0 / 0 / 0 |
| standard | Bundesliga 2 | over35 | 1,166 | 868 | 932 | +64 | 823 / 0 / 0 / 64 |
| standard | Dutch Eredivisie | btts | 1,251 | 1,011 | 1,011 | +0 | 0 / 883 / 76 / 4 |
| standard | Dutch Eredivisie | ht_over05 | 698 | 27 | 130 | +103 | 0 / 0 / 103 / 0 |
| standard | Dutch Eredivisie | ht_over15 | 698 | 31 | 130 | +99 | 0 / 0 / 99 / 0 |
| standard | Dutch Eredivisie | over15 | 1,251 | 850 | 1,000 | +150 | 804 / 107 / 6 / 37 |
| standard | Dutch Eredivisie | over25 | 1,251 | 1,205 | 1,205 | +0 | 0 / 0 / 0 / 0 |
| standard | Dutch Eredivisie | over35 | 1,251 | 904 | 988 | +84 | 855 / 0 / 23 / 61 |
| standard | Greek Super League | btts | 923 | 614 | 614 | +0 | 0 / 489 / 84 / 6 |
| standard | Greek Super League | ht_over05 | 512 | 29 | 29 | +0 | 0 / 0 / 0 / 0 |
| standard | Greek Super League | ht_over15 | 512 | 30 | 30 | +0 | 0 / 0 / 0 / 0 |
| standard | Greek Super League | over15 | 923 | 516 | 597 | +81 | 468 / 50 / 0 / 31 |
| standard | Greek Super League | over25 | 923 | 912 | 912 | +0 | 0 / 0 / 0 / 0 |
| standard | Greek Super League | over35 | 923 | 515 | 566 | +51 | 477 / 0 / 0 / 51 |
| standard | La Liga 2 | btts | 1,832 | 1,369 | 1,369 | +0 | 0 / 1089 / 183 / 19 |
| standard | La Liga 2 | ht_over05 | 1,009 | 57 | 57 | +0 | 0 / 0 / 0 / 0 |
| standard | La Liga 2 | ht_over15 | 1,009 | 59 | 59 | +0 | 0 / 0 / 0 / 0 |
| standard | La Liga 2 | over15 | 1,832 | 1,218 | 1,347 | +129 | 1134 / 68 / 0 / 61 |
| standard | La Liga 2 | over25 | 1,832 | 1,813 | 1,813 | +0 | 6 / 0 / 0 / 0 |
| standard | La Liga 2 | over35 | 1,832 | 1,218 | 1,336 | +118 | 1123 / 0 / 0 / 118 |
| standard | Ligue 2 | btts | 1,327 | 812 | 812 | +0 | 0 / 616 / 179 / 0 |
| standard | Ligue 2 | ht_over05 | 687 | 6 | 6 | +0 | 0 / 0 / 0 / 0 |
| standard | Ligue 2 | ht_over15 | 687 | 11 | 11 | +0 | 0 / 0 / 0 / 0 |
| standard | Ligue 2 | over15 | 1,327 | 786 | 841 | +55 | 764 / 55 / 0 / 0 |
| standard | Ligue 2 | over25 | 1,327 | 1,315 | 1,315 | +0 | 1 / 0 / 0 / 0 |
| standard | Ligue 2 | over35 | 1,327 | 785 | 818 | +33 | 764 / 0 / 0 / 33 |
| standard | Portuguese Primeira Liga | btts | 1,206 | 838 | 838 | +0 | 0 / 711 / 65 / 19 |
| standard | Portuguese Primeira Liga | ht_over05 | 670 | 22 | 22 | +0 | 0 / 0 / 0 / 0 |
| standard | Portuguese Primeira Liga | ht_over15 | 670 | 31 | 31 | +0 | 0 / 0 / 0 / 0 |
| standard | Portuguese Primeira Liga | over15 | 1,206 | 748 | 840 | +92 | 708 / 61 / 0 / 31 |
| standard | Portuguese Primeira Liga | over25 | 1,206 | 1,201 | 1,201 | +0 | 0 / 0 / 0 / 0 |
| standard | Portuguese Primeira Liga | over35 | 1,206 | 757 | 831 | +74 | 716 / 0 / 0 / 74 |
| standard | Scottish Championship | btts | 731 | 22 | 22 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Championship | ht_over05 | 412 | 11 | 11 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Championship | ht_over15 | 412 | 12 | 12 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Championship | over15 | 731 | 20 | 20 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Championship | over25 | 731 | 710 | 710 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Championship | over35 | 731 | 26 | 26 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League One | btts | 730 | 8 | 8 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League One | ht_over05 | 408 | 5 | 5 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League One | ht_over15 | 408 | 9 | 9 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League One | over15 | 730 | 11 | 11 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League One | over25 | 730 | 706 | 706 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League One | over35 | 730 | 9 | 9 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League Two | btts | 718 | 8 | 8 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League Two | ht_over05 | 399 | 5 | 5 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League Two | ht_over15 | 399 | 6 | 6 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League Two | over15 | 718 | 6 | 6 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League Two | over25 | 718 | 713 | 713 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish League Two | over35 | 718 | 9 | 9 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Premiership | btts | 912 | 694 | 694 | +0 | 0 / 563 / 100 / 0 |
| standard | Scottish Premiership | ht_over05 | 512 | 25 | 25 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Premiership | ht_over15 | 512 | 20 | 20 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Premiership | over15 | 912 | 484 | 680 | +196 | 461 / 131 / 0 / 65 |
| standard | Scottish Premiership | over25 | 912 | 893 | 893 | +0 | 0 / 0 / 0 / 0 |
| standard | Scottish Premiership | over35 | 912 | 491 | 658 | +167 | 462 / 0 / 0 / 167 |
| standard | Serie B | btts | 1,522 | 1,001 | 1,001 | +0 | 0 / 757 / 175 / 7 |
| standard | Serie B | ht_over05 | 818 | 52 | 52 | +0 | 0 / 0 / 0 / 0 |
| standard | Serie B | ht_over15 | 818 | 58 | 58 | +0 | 0 / 0 / 0 / 0 |
| standard | Serie B | over15 | 1,522 | 848 | 991 | +143 | 771 / 60 / 0 / 83 |
| standard | Serie B | over25 | 1,522 | 1,488 | 1,488 | +0 | 5 / 0 / 0 / 0 |
| standard | Serie B | over35 | 1,522 | 847 | 984 | +137 | 776 / 0 / 0 / 137 |
| standard | Turkish Super Lig | btts | 1,305 | 943 | 943 | +0 | 0 / 0 / 863 / 38 |
| standard | Turkish Super Lig | ht_over05 | 705 | 24 | 24 | +0 | 0 / 0 / 0 / 0 |
| standard | Turkish Super Lig | ht_over15 | 705 | 26 | 26 | +0 | 0 / 0 / 0 / 0 |
| standard | Turkish Super Lig | over15 | 1,305 | 799 | 909 | +110 | 756 / 0 / 0 / 110 |
| standard | Turkish Super Lig | over25 | 1,305 | 1,305 | 1,305 | +0 | 0 / 0 / 0 / 0 |
| standard | Turkish Super Lig | over35 | 1,305 | 815 | 909 | +94 | 763 / 0 / 0 / 94 |
