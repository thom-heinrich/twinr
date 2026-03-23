# Twinr Labeling Handbook

This handbook defines the user-centered labeling scheme for Twinr's two-stage
local router.

Reference date: `2026-03-22`

## Label order

Apply the labels in this order:

1. `MACHEN_ODER_PRUEFEN`
   Use this if Twinr must do something or inspect live house/device/system state.
2. `PERSOENLICH`
   Use this if the answer depends on the user, the household, earlier Twinr context,
   or stored personal facts.
3. `NACHSCHAUEN`
   Use this if the answer needs current external verification, live world state,
   or safety-sensitive checking.
4. `WISSEN`
   Use this if Twinr can answer from stable general knowledge alone.

## Backend mapping

- `WISSEN` -> backend `parametric`
- `NACHSCHAUEN` -> backend `web`
- `PERSOENLICH` -> backend `memory` by default, backend `tool` when Twinr must inspect
  a live personal calendar/task/reminder/system record
- `MACHEN_ODER_PRUEFEN` -> backend `tool`

## Boundary rules

- `Wie funktioniert ein Timer?` -> `WISSEN`
- `Stell einen Timer.` -> `MACHEN_ODER_PRUEFEN`
- `Was habe ich heute fuer Termine?` -> `PERSOENLICH`
  Backend note: often `tool`, because Twinr may need to inspect a live calendar source.
- `Was ist im Haus los?` -> `MACHEN_ODER_PRUEFEN`
- `Wer ist derzeit Bundeskanzler?` -> `NACHSCHAUEN`
- `Wer war Konrad Adenauer?` -> `WISSEN`
- `Was mag Anna?` -> `PERSOENLICH`
- `Kann ich Ibuprofen mit Blutverduennern nehmen?` -> `NACHSCHAUEN`

## WISSEN

Use for stable explanation, translation, history, science, health background, and
general "how does this work" questions that do not need fresh verification.

1. `Was ist Photosynthese?` | `Erklaer mir Arthrose.` | `Wie funktioniert Schwerkraft?` | `Was bedeutet Inflation?`
2. `Was ist Demenz?` | `Erklaer mir Solarenergie.` | `Wie funktioniert WLAN?` | `Was bedeutet Cholesterin?`
3. `Was ist Osteoporose?` | `Erklaer mir Diabetes Typ zwei.` | `Wie funktioniert Verdauung?` | `Was bedeutet Glukose?`
4. `Was ist eine Waermepumpe?` | `Erklaer mir Parkinson.` | `Wie funktioniert Datensicherung?` | `Was bedeutet Staatsverschuldung?`
5. `Was ist Quantencomputing?` | `Erklaer mir Lungenentzuendung.` | `Wie funktioniert ein Impfstoff?` | `Was bedeutet Blutdruck?`
6. `Was ist der Unterschied zwischen Arthrose und Arthritis?` | `Vergleich Vitamin D und Calcium.` | `Woran erkennt man den Unterschied zwischen Bakterien und Viren?` | `Vergleich HDMI und USB C.`
7. `Was ist der Unterschied zwischen Herzinfarkt und Schlaganfall?` | `Vergleich Erkaeltung und Grippe.` | `Woran erkennt man den Unterschied zwischen Demenz und Delir?` | `Vergleich WLAN und Bluetooth.`
8. `Was ist der Unterschied zwischen Rente und Pension?` | `Vergleich Insulin und Blutzucker.` | `Woran erkennt man den Unterschied zwischen Tablette und Kapsel?` | `Vergleich Nebenkosten und Miete.`
9. `Wie kann man einen Timer einstellen?` | `Erklaer mir, wie man einen Brief formuliert.` | `Wie kann man Nudeln kochen?` | `Erklaer mir, wie man Blutdruck misst.`
10. `Wie kann man einen Videocall vorbereiten?` | `Erklaer mir, wie man eine Pflanze umtopft.` | `Wie kann man eine E Mail beantworten?` | `Erklaer mir, wie man eine Einkaufsliste schreibt.`
11. `Wie kann man das Smartphone lauter stellen?` | `Erklaer mir, wie man die Heizung entlueftet.` | `Wie kann man einen Verband wechseln?` | `Erklaer mir, wie man ein Passwort sicher notiert.`
12. `Wer war Marie Curie?` | `Warum ist Sophie Scholl bekannt?` | `Wann lebte Ada Lovelace?` | `Wer war Konrad Adenauer?`
13. `Warum ist Albert Einstein bekannt?` | `Wann lebte Johann Sebastian Bach?` | `Wer war Rosalind Franklin?` | `Warum ist Hildegard von Bingen bekannt?`
14. `Wann lebte Clara Schumann?` | `Wer war Otto von Bismarck?` | `Warum ist Alan Turing bekannt?` | `Wann lebte Nelson Mandela?`
15. `Wie sagt man Guten Morgen auf Italienisch?` | `Was heisst Vielen Dank auf Spanisch?` | `Uebersetz Wie geht es dir ins Englische.` | `Wie sagt man Ich brauche Hilfe auf Franzoesisch?`
16. `Was heisst Bitte langsam sprechen auf Portugiesisch?` | `Uebersetz Wo ist der Bahnhof ins Niederlaendische.` | `Wie sagt man Ich moechte einen Kaffee auf Englisch?` | `Was heisst Bis spaeter auf Italienisch?`
17. `Erklaer mir kurz, wie ein Regenbogen entsteht.` | `Warum rostet Eisen?` | `Wie entsteht Schnee?` | `Warum schlafen Menschen?`
18. `Wie funktioniert ein Herzschrittmacher?` | `Was ist ein Antivirusprogramm?` | `Erklaer mir, was ein Router ist.` | `Wie funktioniert Bluetooth?`
19. `Was ist ein Schlagwort in der Politik?` | `Warum steigt der Blutdruck bei Stress?` | `Wie funktioniert die Verdunstung?` | `Was bedeutet Kalorienbedarf?`
20. `Wofuer ist Vitamin B12 wichtig?` | `Was macht die Schilddruese?` | `Warum braucht man Schlaf?` | `Wie funktioniert eine Brille?`
21. `Was ist der Unterschied zwischen Kaffee und Espresso?` | `Was ist der Unterschied zwischen PDF und Word?` | `Vergleich Gasheizung und Fernwaerme.` | `Vergleich Tablet und Laptop.`
22. `Wie kann man eine Telefonnummer aufschreiben, damit man sie gut lesen kann?` | `Wie formuliert man eine kurze Entschuldigung?` | `Wie kann man ein Hoergeraet reinigen?` | `Wie kann man einen Spaziergang planen?`
23. `Was ist der Unterschied zwischen Erinnerung und Alarm?` | `Wie funktioniert ein Kalender?` | `Erklaer mir, was ein WLAN Passwort ist.` | `Wie funktioniert mobiles Internet?`
24. `Warum ist Wassertrinken wichtig?` | `Was passiert bei einer Impfung?` | `Erklaer mir, warum der Himmel blau ist.` | `Wie funktioniert ein Blutdruckmessgeraet?`
25. `Was ist eine Demokratie?` | `Warum altern Menschen?` | `Wie funktioniert ein Aufzug?` | `Erklaer mir, was Recycling bedeutet.`

## NACHSCHAUEN

Use for live world state, recent events, prices, schedules, warnings, and
questions that should be externally verified before answering.

1. `Wie wird das Wetter heute in Berlin?` | `Brauche ich morgen in Hamburg einen Schirm?` | `Wie warm wird es heute Abend in Muenchen?` | `Wie wird das Wetter am Wochenende in Koeln?`
2. `Wie wird das Wetter heute Nacht in Leipzig?` | `Brauche ich morgen Mittag in Bremen einen Schirm?` | `Wie warm wird es heute in Dresden?` | `Wie wird das Wetter morgen in Stuttgart?`
3. `Was ist heute in Deutschland passiert?` | `Gibt es gerade Neuigkeiten aus Berlin?` | `Fass mir die aktuellen Meldungen aus Europa zusammen.` | `Was ist seit heute Morgen in Hamburg passiert?`
4. `Gibt es im Moment Neuigkeiten zur Bahn?` | `Fass mir die aktuellen Meldungen zu Energiepreisen zusammen.` | `Was ist heute in Brandenburg passiert?` | `Gibt es aktuelle Nachrichten zum Gesundheitssystem?`
5. `Was ist heute beim Bundestag passiert?` | `Gibt es gerade Neuigkeiten zu Renten?` | `Fass mir die aktuellen Meldungen zum Nahverkehr zusammen.` | `Was ist in den letzten Stunden in Sachsen passiert?`
6. `Wann faehrt der naechste Zug von Berlin nach Potsdam?` | `Wann faehrt der naechste Zug von Hamburg nach Hannover?` | `Wann faehrt der naechste Zug von Koeln nach Bremen?` | `Wann faehrt der naechste Zug von Dresden nach Jena?`
7. `Hat die Apotheke am Markt jetzt geoeffnet?` | `Hat das Rathaus heute Nachmittag geoeffnet?` | `Hat die Bibliothek morgen frueh geoeffnet?` | `Hat der Supermarkt um die Ecke am Sonntag geoeffnet?`
8. `Ist die A9 gerade gesperrt?` | `Ist die Ringbahn heute Abend gesperrt?` | `Ist die Strecke nach Dresden im Moment gesperrt?` | `Ist die Linie U2 morgen frueh gesperrt?`
9. `Wie teuer ist Gold gerade?` | `Wie hoch ist aktuell der Preis von Bitcoin?` | `Wie teuer ist Strom im Moment?` | `Wie hoch ist aktuell der Preis von Gas?`
10. `Ist das Diabetes Messgeraet im Moment lieferbar?` | `Ist der Wasserfilter gerade lieferbar?` | `Ist die Tintenpatrone aktuell verfuegbar?` | `Ist der Blutdrucksensor im Moment lieferbar?`
11. `Kann ich Ibuprofen mit Blutverduennern zusammen nehmen?` | `Kann ich Aspirin mit Schlafmitteln zusammen nehmen?` | `Kann ich Paracetamol mit Antibiotika zusammen nehmen?` | `Kann ich Johanniskraut mit Blutdrucksenkern zusammen nehmen?`
12. `Gibt es aktuell Rueckrufe fuer Raeucherlachs?` | `Gibt es aktuell Rueckrufe fuer Babynahrung?` | `Gibt es aktuell Rueckrufe fuer Fertigsalat?` | `Gibt es aktuell Rueckrufe fuer Blutdrucktabletten?`
13. `Gibt es heute eine Warnung vor Glatteis in Berlin?` | `Gibt es gerade eine Warnung vor Hochwasser in Sachsen?` | `Gibt es fuer morgen eine Warnung vor Gewitter in Bayern?` | `Gibt es im Moment eine Warnung vor starker Hitze in Hessen?`
14. `Wer ist derzeit Bundeskanzler?` | `Wer ist aktuell Praesident in Frankreich?` | `Wer ist im Moment Chef der Deutschen Bahn?` | `Wer fuehrt derzeit die Tabelle in der Bundesliga?`
15. `Wie ist der Dollar Kurs heute?` | `Wie hoch ist der Benzinpreis gerade?` | `Wie teuer ist Heizöl aktuell?` | `Wie hoch ist der Silberpreis im Moment?`
16. `Was machen die Nachrichten gerade mit den Energiepreisen?` | `Gibt es neue Meldungen zum Pflegenotstand?` | `Was ist heute zum Stromnetz passiert?` | `Gibt es aktuelle Neuigkeiten zum Flughafen?`
17. `Wie ist das Wetter derzeit in Rom?` | `Wird es heute in Wien regnen?` | `Wie warm ist es gerade in Hamburg?` | `Gibt es heute Abend Sturm in Rostock?`
18. `Hat die Sparkasse am Platz heute geoeffnet?` | `Hat die Physiopraxis morgen geoeffnet?` | `Hat das Hallenbad heute offen?` | `Hat der Buergerservice naechste Woche geoeffnet?`
19. `Wann kommt der naechste Bus zum Bahnhof?` | `Wann faehrt die naechste Regionalbahn nach Magdeburg?` | `Wann startet der naechste Flug nach Wien?` | `Wann kommt die naechste Faehre?`
20. `Wie ist die Verkehrslage auf der A7 gerade?` | `Ist die S Bahn zum Flughafen aktuell verspätet?` | `Wie ist die Lage auf der Strecke nach Potsdam?` | `Gibt es Stau auf dem Ring im Moment?`
21. `Was ist heute an der Boerse passiert?` | `Wie stehen die Aktien von Siemens aktuell?` | `Wie entwickelt sich Bitcoin gerade?` | `Was ist heute an den Maerkten los?`
22. `Gibt es gerade neue Richtlinien fuer Pflegegeld?` | `Was hat sich heute bei der Rentenpolitik geaendert?` | `Gibt es derzeit neue Einreisebestimmungen?` | `Welche Bahnstreiks laufen aktuell?`
23. `Ist derzeit eine Unwetterwarnung fuer Brandenburg aktiv?` | `Wie ist die Feinstaublage gerade in Berlin?` | `Gibt es aktuell Waldbrandwarnungen in Thueringen?` | `Ist die Luftqualitaet heute schlecht?`
24. `Wie lange wartet man heute im Buergerservice?` | `Wie voll ist der Weihnachtsmarkt gerade?` | `Gibt es heute eine Demo in der Innenstadt?` | `Ist die Notaufnahme derzeit ueberlastet?`
25. `Welche Nachrichten sind momentan wichtig?` | `Was ist gerade das groesste Thema in Deutschland?` | `Was wird heute ueber Pflege diskutiert?` | `Welche Meldungen sind heute Abend neu dazugekommen?`

## PERSOENLICH

Use for user-specific facts, household facts, remembered preferences, earlier
Twinr context, and live personal lookups such as the user's calendar or open
reminders.

Memory-backed examples:

1. `Wie heisst meine Enkelin?` | `Wann hat mein Enkel Geburtstag?` | `Wie heisst mein Hausarzt?` | `Wie heisst meine Pflegerin?`
2. `Wie heisst mein Kardiologe?` | `Wann hat meine Tochter Geburtstag?` | `Wie heisst mein Physiotherapeut?` | `Wie heisst mein Zahnarzt?`
3. `Welchen Kuchen mag ich am liebsten?` | `Welchen Tee trinke ich abends gern?` | `Welche Musik hoere ich gern, wenn ich unruhig bin?` | `Welche Suppe mag ich besonders?`
4. `Wo haben wir den Zweitschluessel hingelegt?` | `Wo liegt die Taschenlampe normalerweise?` | `In welchem Schrank bewahren wir die Notfallmappe auf?` | `Wo liegt der Reisepass?`
5. `Was hatten wir ueber den Urlaub festgehalten?` | `Woran wollte ich mich morgen erinnern?` | `Welchen Plan hatten wir fuer Ostern?` | `Was hatten wir fuer den Arztbesuch notiert?`
6. `Welche Allergien habe ich?` | `Welche Tabletten nehme ich morgens?` | `Was steht in meinen Notizen zu meinem Blutdruck?` | `Was steht in meinen Notizen zu meinem Ruecken?`
7. `Welche Telefonnummer hat meine Tochter?` | `Welche Adresse habe ich fuer den Pflegedienst gespeichert?` | `Welche Kontaktdaten habe ich zu Frau Schneider notiert?` | `Welche Telefonnummer hat mein Hausarzt?`
8. `Wann mache ich normalerweise die Gymnastik?` | `An welchem Tag ist bei mir der Wocheneinkauf dran?` | `Zu welcher Uhrzeit mache ich meistens den Spaziergang?` | `Wann bringe ich normalerweise den Muell raus?`
9. `Was mag Anna besonders gern?` | `Worueber freut sich Karl am meisten?` | `Was hatten wir ueber Frau Schneider festgehalten?` | `Was mag mein Enkel gern essen?`
10. `Was mag meine Tochter besonders?` | `Worueber freut sich mein Sohn?` | `Was hatten wir ueber den Pfleger Jens notiert?` | `Was mag meine Nachbarin gern trinken?`
11. `Wie heisst mein Fahrdienst?` | `Wie heisst mein Friseur?` | `Wie heisst mein Orthopaede?` | `Wie heisst mein Podologe?`
12. `Wo liegt das Ladegeraet?` | `Wo bewahren wir die Ersatzbrille auf?` | `Wo liegt das Rezeptheft?` | `Wo sind die Winterdecken verstaut?`
13. `Was wollten wir fuer die Geburtstagsfeier besorgen?` | `Was hatten wir fuer den Einkauf festgehalten?` | `Welchen Plan hatten wir fuer die Reise nach Hamburg?` | `Was war fuer den Chorabend notiert?`
14. `Was steht in meinen Notizen zu meinem Schlaf?` | `Was steht in meinen Notizen zu meinem Magen?` | `Was steht in meinen Notizen zu meinem Blutzucker?` | `Was steht in meinen Notizen zu meiner Bewegung?`
15. `Welche Telefonnummer hat meine Schwester?` | `Welche Kontaktdaten habe ich zu meinem Nachbarn?` | `Welche Adresse habe ich fuer meinen Physiotherapeuten?` | `Welche Telefonnummer hat die Apotheke?`

Tool-backed personal lookup examples:

16. `Was habe ich heute fuer Termine?` | `Was habe ich morgen fuer Termine?` | `Was steht heute in meinem Kalender?` | `Was steht naechste Woche in meinem Kalender?`
17. `Kommt heute jemand zu mir?` | `Kommt morgen jemand zu mir?` | `Ist fuer heute Besuch eingetragen?` | `Ist naechste Woche jemand bei mir eingetragen?`
18. `Habe ich heute noch Medikamente offen?` | `Habe ich heute Abend noch Medikamente offen?` | `Habe ich morgen Medikamente offen?` | `Habe ich fuer heute noch Erinnerungen offen?`
19. `Welche Erinnerung habe ich heute Abend?` | `Welche Erinnerung habe ich morgen?` | `Welche Erinnerung habe ich fuer heute?` | `Welche Erinnerung habe ich naechste Woche?`
20. `Welche Aufgaben habe ich heute noch offen?` | `Welche Aufgaben habe ich morgen offen?` | `Welche Aufgaben habe ich fuer heute offen?` | `Welche Aufgaben habe ich diese Woche offen?`
21. `Wann kommt der Pflegedienst zu mir?` | `Wann kommt der Hausarzt zu mir?` | `Wann kommt der Fahrdienst?` | `Wann kommt meine Physiotherapie?`
22. `Was steht heute bei mir an?` | `Was steht morgen bei mir an?` | `Was steht heute Nachmittag bei mir an?` | `Was steht heute Abend bei mir an?`
23. `Habe ich heute einen Arzttermin?` | `Habe ich morgen einen Zahnarzttermin?` | `Habe ich diese Woche einen Termin?` | `Habe ich am Montag einen Termin?`
24. `Was ist meine naechste Erinnerung?` | `Was ist mein naechster Termin?` | `Was ist meine naechste Aufgabe?` | `Was ist meine naechste Medikamentenerinnerung?`
25. `Ist fuer morgen ein Besuch eingetragen?` | `Ist fuer heute Abend ein Termin eingetragen?` | `Ist fuer Montag etwas in meinem Kalender?` | `Ist fuer naechste Woche ein Termin bei mir geplant?`

## MACHEN_ODER_PRUEFEN

Use for explicit actions, device control, house-state inspection, opening views,
messaging, calling, printing, timers, alarms, and other operational requests.

1. `Stell einen Timer auf 10 Minuten.` | `Stell einen Timer auf 20 Minuten.` | `Stell einen Timer auf 45 Minuten.` | `Stell einen Timer auf eine Stunde.`
2. `Mach mir einen Wecker fuer 7 Uhr.` | `Mach mir einen Wecker fuer morgen um 9.` | `Mach mir einen Wecker fuer Freitag um 11.` | `Mach mir einen Wecker fuer heute Nacht um 23 Uhr.`
3. `Erinnere mich heute Abend an die Tabletten.` | `Erinnere mich morgen frueh an den Anruf bei meiner Tochter.` | `Erinnere mich uebermorgen an den Blutdruck.` | `Erinnere mich am Wochenende an das Blumen giessen.`
4. `Druck meine Einkaufsliste.` | `Druck den Arzttermin fuer Freitag.` | `Druck die Notiz fuer den Hausarzt.` | `Druck die Adresse von Frau Schneider.`
5. `Schreib Milch und Brot auf die Liste.` | `Schreib meine Medikamente fuer morgen auf.` | `Schreib den Einkaufszettel fuer Ostern auf.` | `Schreib die Telefonnummer vom Pflegedienst auf.`
6. `Notier dir bitte den Arzttermin.` | `Notier dir bitte meine To do Liste fuer heute.` | `Notier dir bitte die Rezepte fuer diese Woche.` | `Notier dir bitte die Adresse vom Fahrdienst.`
7. `Mach bitte lauter.` | `Stell die Lautstaerke leiser.` | `Starte das Radio.` | `Stoppe das Hoerbuch.`
8. `Starte meine Abendmusik.` | `Starte die Entspannungsmusik.` | `Stoppe die Nachrichten.` | `Starte meine Lieblingsplaylist.`
9. `Pruef, ob der Drucker verbunden ist.` | `Pruef, ob das Mikrofon verbunden ist.` | `Pruef, ob die Kopfhoerer verbunden sind.` | `Pruef, ob der Router verbunden ist.`
10. `Schau nach, ob der Drucker Papier hat.` | `Schau nach, ob das WLAN stabil ist.` | `Schau nach, ob eine Druckwarteschlange haengt.` | `Schau nach, ob der Wecker aktiv ist.`
11. `Zeig mir die heutigen Termine.` | `Zeig mir die letzten Notizen.` | `Zeig mir die Einkaufsliste.` | `Zeig mir die Wecker fuer diese Woche.`
12. `Schick meiner Tochter die Nachricht ich komme spaeter.` | `Schick meinem Sohn die Nachricht bitte bring Brot mit.` | `Schick Frau Schneider die Nachricht ich denke an dich.` | `Schick dem Pflegedienst die Nachricht der Termin ist bestaetigt.`
13. `Ruf bitte meine Schwester an.` | `Ruf bitte den Hausarzt an.` | `Ruf bitte den Fahrdienst an.` | `Ruf bitte meinen Enkel an.`
14. `Oeffne meine Erinnerungen.` | `Oeffne den Kalender.` | `Oeffne die Notizen.` | `Oeffne die Druckwarteschlange.`
15. `Was ist im Haus los?` | `Ist die Haustuer gerade offen?` | `Ist das Kuechenfenster gerade offen?` | `Wie ist der Status im Wohnzimmer?`
16. `Ist die Balkontuer offen?` | `Ist das Schlafzimmerfenster offen?` | `Wie ist der Status in der Kueche?` | `Wie ist der Status im Flur?`
17. `Schalte das Licht ein.` | `Schalte den Fernseher aus.` | `Mach den Ventilator an.` | `Stell die Heizung niedriger.`
18. `Mach die Lampe im Wohnzimmer an.` | `Mach die Steckdose aus.` | `Starte den Staubsaugerroboter.` | `Stoppe den Staubsaugerroboter.`
19. `Pruef, ob jemand geklingelt hat.` | `Pruef, ob die Waschmaschine fertig ist.` | `Pruef, ob der Trockner laeuft.` | `Pruef, ob die Haustuer abgeschlossen ist.`
20. `Zeig mir den Status vom Drucker.` | `Zeig mir meine offenen Aufgaben.` | `Zeig mir den Status vom WLAN.` | `Zeig mir die naechste Erinnerung.`
21. `Druck die Einkaufsliste fuer morgen.` | `Druck den Wochenplan.` | `Druck die Medikamentenliste.` | `Druck die Telefonnummern fuer den Notfall.`
22. `Schick meinem Enkel die Nachricht wir sehen uns am Sonntag.` | `Schick dem Nachbarn die Nachricht ich brauche morgen Hilfe.` | `Schick der Apotheke die Nachricht die Medikamente sind bestellt.` | `Schick meinem Sohn die Nachricht ich bin schon unterwegs.`
23. `Ruf den Pflegedienst an.` | `Ruf den Nachbarn an.` | `Ruf meine Tochter an.` | `Ruf Frau Schneider an.`
24. `Mach bitte das Display heller.` | `Starte die Wettervorhersage.` | `Stoppe die Musik.` | `Spiel die klassische Musik.`
25. `Schau nach, ob eine neue Nachricht angekommen ist.` | `Schau nach, ob die Einkaufsliste leer ist.` | `Schau nach, ob heute noch Erinnerungen offen sind.` | `Schau nach, ob der Bildschirm verbunden ist.`
