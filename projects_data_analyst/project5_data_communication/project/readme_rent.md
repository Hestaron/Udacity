# Exploration of rental transactions in the Netherlands
## by Yannick Mariman


## Dataset

> In the dataset are 534.000 transactions with 15 features. This data is confidential and used during in my work. 
There are 15 features in the dataset:
<li>Bron is the source from which the data is coming.
<li>Bouwjaar is the year in which the building is build.
<li>M2HuurPrijs is the square metre price for renting.
<li>GebruiksOppervlakte is the usable square metres in a house.
<li>AanmeldDatum is the date at which the building is put online.
<li>Postcode is zipcode/postalcode.
<li>TypeWoning is the type of house people live in.
<li>Looptijd is the time between putting the house online and the signing of the contract.
<li>TransactieHuurPrijs is the total amount of euros paid.
<li>TransactieDatumOndertekeningAkte is the date that the contract is signed.
<li>OnderhoudsNiveaBinnen is the maintenance level inside of the building.
<li>OnderhoudsNiveaBuiten is the maintenance level outside of the building.
<li>GemeenteNaam is the name of the municipality.
<li>GemeenteCat is the category of municipality. The big four, number 5 until 45 and the rest.



OnderhoudsNiveauBinnen, OnderhoudsNiveauBuiten and Energielabels are ordinal features. They are sorted from 'bad' to 'better'. In the case of GemeenteCat (Municipality Category) it is sorted from bigger municipality category to smaller.
'EnergieLabel': ['G','F','E','D','C', 'B', 'A']
'OnderhoudsNiveauBinnen': ['Slecht', 'Slecht tot matig', 'Matig', 'Matig tot redelijk', 'Redelijk', 'Redelijk tot goed', 'Goed', 'Goed tot uitstekend', 'Uitstekend']
'OnderhoudsNiveauBuiten': ['Slecht', 'Slecht tot matig', 'Matig', 'Matig tot redelijk', 'Redelijk', 'Redelijk tot goed', 'Goed', 'Goed tot uitstekend', 'Uitstekend']
'GemeenteCat': ['G4', 'G40', 'Overig']


## Summary of Findings

> Maintenance levels inside and outside follow roughly the same distribution in different municipality categories. That is an interesting find. I plan on using this exploration path for the presentation.
> Also Energy labels for different municipality categories staying roughly the same is an interesting find.


## Key Insights for Presentation

> The exploration path follows the different maintenance levels for different municipality categories.