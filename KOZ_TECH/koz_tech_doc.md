# Közösségi és technológiai hálózatok - Dáné Bence

## Adathalmaz és alap gráf információk
A projekt a [Cora](https://graphsandnetworks.com/the-cora-dataset/), tudományos publikációkból álló adathalmazt használja.

<table>
  <tr>
    <th>N_nodes</th>
    <th>N_edges</th>
    <th>Avg<br>degree</th>
    <th>Density</th>
    <th>Avg<br>clustering</th>
    <th>Connected<br>components</th>
    <th>Giant<br>component<br>size</th>
    <th>Diameter</th>
    <th>Avg<br>shortest<br>path</th>
  </tr>
  <tr>
    <td>2708</td>
    <td>5278</td>
    <td>3.9</td>
    <td>0</td>
    <td>0.24</td>
    <td>78</td>
    <td>2485</td>
    <td>19</td>
    <td>6.31</td>
  </tr>
</table>

A gráf "hiányos", olyan értelemben, hogy a lehetséges összes él helyett, mindössze 5278 van. Ezt támasztja alá a 3.9-es átlag fokszám, illetve, 78 összefüggő komponens található, de a node-ok nagytöbbsége a giant componentbe tartozik.

<div style="display: flex; justify-content: center;">
  <img src="https://github.com/user-attachments/assets/07fcd076-b511-49f5-840e-d3358d4ba031" width="45%">
  <img src="https://github.com/user-attachments/assets/541374be-7dc3-4d38-8eae-be95aa38efc5" width="45%">
</div>
Egy csúcs összes szomszédja ugyan azon osztálycímkét viseli, ami nem meglepő, hiszen a kutatási cikkek rendszerint a saját területükhöz tartozó cikkekre hivatkoznak.


<div align="center">
  <table>
    <tr>
      <th>
        Avg_triangles
      </th>   
      <th>
        Max_core
      </th>
    </tr>
        <tr>
      <td>
        1.8
      </td>   
      <td>
        4
      </td>
    </tr>
    
  </table>
  <img src="https://github.com/user-attachments/assets/57482b4d-3fe9-45e4-9ae9-66bd0bebed7b" width="65%">
</div>

A fenti ábra négyféle szerepkör alapján kategorizálja a csomópontokat:
- Hub – kiemelkedően nagy fokszámú csúcsok (top 5%)
- Bridge – magas átfedési központiságú (betweenness) csúcsok (top 5%)
- Peripheral – alacsony PageRank értékű csúcsok (bottom 10%)
- Normal – minden más csúcs

## Közösségek és GCN

<div align="center">
<table>
  <tr>
    <th>Mérőszám</th>
    <th>Érték</th>
  </tr>
  <tr>
    <td><strong>Közösségek száma</strong></td>
    <td>105</td>
  </tr>
  <tr>
    <td><strong>Modularitás</strong></td>
    <td>0.8106</td>
  </tr>
  <tr>
    <td><strong>NMI (címkék vs Louvain)</strong></td>
    <td>0.454</td>
  </tr>
  <tr>
    <td><strong>ARI (címkék vs Louvain)</strong></td>
    <td>0.251</td>
  </tr>
</table>
  <img alt="labels_vs_communities" src="https://github.com/user-attachments/assets/70f93230-95a0-4f4f-9773-b4177e8f1a2f" width="85%"/>

</div>

A gráfban amúgy 7 valódi címke található, viszont a Louvain sokkal több, 105 közösséget talált. A modularitás jelzi, hogy ezek jól elkülöníthetők topológiailag, viszont mint az NMI és ARI metrikák alapján ezek nem egyeznek a csúcsok valódi címkéivel.
### GCN
2 layer-ből álló GCN:
```python
class GCN(nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        return self.conv2(x, edge_index)
```
<div style="display: flex; justify-content: center;">
<img alt="embs" src="https://github.com/user-attachments/assets/8439d950-8372-46f4-acad-18c872933dff" width="45%"/>
<img alt="decision_boundary" src="https://github.com/user-attachments/assets/b84ba24b-7f94-4dc0-afe0-4302d5d38a78" width="45%"/>
</div>
A közösségek köré rajzolt konvex burkok (eredeti, embeddings):
<div style="display: flex; justify-content: center;">
  <img alt="original_feature_hulls" src="https://github.com/user-attachments/assets/a40592ad-6917-42d1-85ad-3f5f98fbb7d8" width="45%"/>
  <img alt="gnn_embedding_hulls" src="https://github.com/user-attachments/assets/778dc5e0-afd2-4c37-b544-a02810016333"  width="45%"/>
</div >
A  végső beágyazásook alapján egy k-legközelebbi szomszéd (k-NN) gráfot készítünk, amely nem az eredeti topológiát, hanem a tanult reprezentációkat tükrözi.
Az embedding gráf sűrűbb, mint az eredeti gráf, illetve magasabb klaszterezési együttható jelzi,hogy az embedding térben kompaktabb közösségek.
<div style="display: flex; justify-content: center;">
  <table>
    <tr>
      <th>Mérőszám</th>
      <th>Embedding gráf</th>
      <th>Eredeti gráf</th>
    </tr>
    <tr>
      <td><strong>Átlag fokszám</strong></td>
      <td>7.12</td>
      <td>3.90</td>
    </tr>
    <tr>
      <td><strong>Átlag klaszt. együttható</strong></td>
      <td>0.388</td>
      <td>0.241</td>
    </tr>
  </table>
  Az embedding gráfon Louvain közösségkeresést alkalmazva, majd összevetb a valódi címkékkel:
  <table>
    <tr>
      <th>Mérőszám</th>
      <th>Érték</th>
    </tr>
    <tr>
      <td><strong>NMI (embedding vs címkék)</strong></td>
      <td>0.540</td>
    </tr>
    <tr>
      <td><strong>ARI (embedding vs címkék)</strong></td>
      <td>0.371</td>
    </tr>
  </table>
</div>

<h2> k-NN embedding gráf – GCN által tanult közösségek</h2>

<div align="center">
  <img src="https://github.com/user-attachments/assets/7530e5bf-3ab0-46c1-a6bf-837b518b6087" width="70%">
</div>
<p>
Ez az ábra a GNN végső beágyazásai alapján felépített k-NN gráfot mutatja, ahol a csúcsok színezése a
<strong>beágyazási térben detektált Louvain közösségek</strong> szerint történik.
</p>
<hr>

<h2> k-NN embedding gráf – Valódi címkék szerint színezve</h2>

<div align="center">
  <img src="https://github.com/user-attachments/assets/14b8d691-8d7d-41a4-8c14-e4cd4b8cbd46" width="70%">
</div>

<p>
Az ábrán ugyanaz a k-NN embedding gráf látható, azonban a csúcsok a <strong>valódi osztálycímkék</strong> szerint vannak színezve.
</p>
<hr>

<h2> k-NN embedding gráf – Eredeti gráf Louvain közösségei</h2>

<div align="center">
  <img src="https://github.com/user-attachments/assets/65f7125a-422d-415c-a20d-2d56b693c0c8" width="70%">
</div>

<p>
Ez az ábra azt szemlélteti, hogy a k-NN embedding gráf csúcsai hogyan viszonyulnak az
<strong>eredeti gráf topológiáján detektált Louvain közösségekhez</strong>.
</p>
<hr>

<h2> Közösséget váltó csúcsok az embedding térben</h2>

<div align="center">
  <img src="https://github.com/user-attachments/assets/b2950fbb-ba90-4870-8812-ec7c5465f466" width="70%">
</div>
<p>
Az ábrán pirossal jelölve láthatók azok a csúcsok, amelyek a GNN beágyazási gráfban
<strong>más közösségbe kerültek</strong>, mint az eredeti gráf Louvain közösségei alapján.
</p>

### Beágyazások intra-class és inter-class távolságai
<ul>
  <li>
    <strong>Intra-class távolság:</strong><br>
    az egyes osztályokon belüli csomópontok átlagos távolsága az osztály centroidjától
  </li>
  <li>
    <strong>Inter-class távolság:</strong><br>
    az osztály-centroidok közötti átlagos távolság
  </li>
</ul>

<table>
  <tr>
    <th>Epoch</th>
    <th>Intra-class</th>
    <th>Inter-class</th>
    <th>Inter / Intra</th>
  </tr>

  <tr>
    <td>10</td>
    <td>1.48</td>
    <td>3.72</td>
    <td>2.51</td>
  </tr>

  <tr>
    <td>50</td>
    <td>2.36</td>
    <td>6.09</td>
    <td>2.58</td>
  </tr>

  <tr>
    <td>100</td>
    <td>1.86</td>
    <td>4.87</td>
    <td>2.62</td>
  </tr>

  <tr>
    <td style="text-align:center;">…</td>
    <td style="text-align:center;">…</td>
    <td style="text-align:center;">…</td>
    <td style="text-align:center;">…</td>
  </tr>

  <tr>
    <td>150</td>
    <td>1.82</td>
    <td>4.62</td>
    <td>2.54</td>
  </tr>

  <tr>
    <td>200</td>
    <td>1.84</td>
    <td>4.59</td>
    <td>2.49</td>
  </tr>
</table>
<p>
Az Inter / Intra arány a tanulás középső szakaszában éri el a maximumát, 
ami azt jelzi, ekkor jön létre a legjobban szeparált embedding struktúra. 
</p>


<div align="center">
  <img src="https://github.com/user-attachments/assets/8c8ab2ce-5195-47b9-bbcb-9db163d1e5c6" width="70%">
</div>

### Összehasonlítás
<table>
  <tr>
    <th>Összehasonlítás</th>
    <th>NMI</th>
    <th>ARI</th>
  </tr>
  <tr>
    <td><strong>GCN predikciók vs Louvain közösségek</strong></td>
    <td>0.468</td>
    <td>0.287</td>
  </tr>
  <tr>
    <td><strong>GCN predikciók vs valódi címkék</strong></td>
    <td>0.613</td>
    <td>0.617</td>
  </tr>
</table>

<p>
A GCN által tanult predikciók jobban egyeznek a valódi osztálycímkékkel,
mint a Louvain-alapú közösségstruktúrával.
</p>

<ul>
  <li>
    <strong>GCN vs címkék (NMI = 0.613, ARI = 0.617):</strong><br>
    A modell hatékonyan tanulja meg az osztálystruktúrát.
  </li>
  <li>
    <strong>GCN vs Louvain (NMI = 0.468, ARI = 0.287):</strong><br>
    A Louvain közösségek csak részben fedik le az osztálycímkéket, mivel a módszer
    kizárólag topológiai információt használ.
  </li>
</ul>

## Szimuláció
### Edge dropout
<p>
Az edge dropout szimuláció célja annak vizsgálata volt, hogy a gráf szerkezeti
sérülése (véletlenszerű élek eltávolítása) milyen hatással van a Louvain
közösségdetektálásra és a GCN predikciókra.
</p>

<table>
  <tr>
    <th>Edge dropout arány</th>
    <th>Louvain NMI</th>
    <th>Louvain ARI</th>
    <th>GCN NMI</th>
    <th>GCN ARI</th>
  </tr>
  <tr>
    <td>0%</td>
    <td>0.457</td>
    <td>0.261</td>
    <td>0.504</td>
    <td>0.505</td>
  </tr>
  <tr>
    <td>10%</td>
    <td>0.451</td>
    <td>0.258</td>
    <td>0.494</td>
    <td>0.496</td>
  </tr>
  <tr>
    <td>20%</td>
    <td>0.418</td>
    <td>0.197</td>
    <td>0.488</td>
    <td>0.491</td>
  </tr>
  <tr>
    <td>40%</td>
    <td>0.398</td>
    <td>0.113</td>
    <td>0.471</td>
    <td>0.467</td>
  </tr>
  <tr>
    <td>60%</td>
    <td>0.385</td>
    <td>0.064</td>
    <td>0.461</td>
    <td>0.456</td>
  </tr>
</table>
<div style="display: flex; justify-content: center;">
  <img width="45%" alt="ari_plot" src="https://github.com/user-attachments/assets/cf2e9e90-3e3d-4b05-976c-628ab549c969" />
  <img width="45%" alt="nmi_plot" src="https://github.com/user-attachments/assets/c008eb16-21af-470a-8580-980c99f42587" />
</div >
<ul>
  <li>
    <strong>Louvain:</strong><br>
    A Louvain algoritmus teljesítménye gyorsan romlik az élek eltávolításával.
    Már 20–40% dropout esetén jelentős NMI és ARI csökkenés figyelhető meg,
    mivel az algoritmus kizárólag a topológiára támaszkodik.
  </li>
  <li>
    <strong>GCN:</strong><br>
    A GCN lényegesen robusztusabb: még 60%-os élvesztés mellett is
    viszonylag magas NMI és ARI értékeket tart fenn.
  </li>
</ul>

### Feature noise
<p>
A feature noise szimuláció azt mutatja meg, hogy a node-featureök zajosítása
hogyan befolyásolja a GCN teljesítményét, miközben a Louvain algoritmus
változatlan marad.
</p>
<table>
  <tr>
    <th>Feature noise σ</th>
    <th>GCN NMI</th>
    <th>GCN ARI</th>
    <th>Louvain NMI</th>
    <th>Louvain ARI</th>
  </tr>
  <tr>
    <td>0.0</td>
    <td>0.613</td>
    <td>0.617</td>
    <td>0.453</td>
    <td>0.226</td>
  </tr>
  <tr>
    <td>0.05</td>
    <td>0.612</td>
    <td>0.616</td>
    <td>0.453</td>
    <td>0.226</td>
  </tr>
  <tr>
    <td>0.1</td>
    <td>0.601</td>
    <td>0.600</td>
    <td>0.453</td>
    <td>0.226</td>
  </tr>
  <tr>
    <td>0.2</td>
    <td>0.577</td>
    <td>0.581</td>
    <td>0.453</td>
    <td>0.226</td>
  </tr>
  <tr>
    <td>0.5</td>
    <td>0.422</td>
    <td>0.402</td>
    <td>0.453</td>
    <td>0.226</td>
  </tr>
</table>

<div style="display: flex; justify-content: center;">
  <img width="45%"  alt="ari_feature_noise" src="https://github.com/user-attachments/assets/b674adde-b69b-4b00-a2e3-64a6bb74cef9" />
  <img width="45%"  alt="nmi_feature_noise" src="https://github.com/user-attachments/assets/7ab48672-9c2a-4d44-80af-d6b6fd10f01d" />
</div >

<ul>
  <li>
    <strong>GCN viselkedése:</strong><br>
    Kis zajszinteknél (σ ≤ 0.05) a GCN teljesítménye gyakorlatilag változatlan,
    ami azt jelzi, hogy a modell nem túlérzékeny a kisebb feature-zajokra.
    Nagyobb zaj esetén (σ ≥ 0.2) azonban az NMI és ARI fokozatosan romlik,
    különösen σ = 0.5 mellett.
  </li>
  <li>
    <strong>Louvain viselkedése:</strong><br>
    A Louvain algoritmus eredményei konstansak maradnak minden zajszintnél,
    mivel az algoritmus kizárólag a gráf topológiáját használja,
    és teljesen független a feature-öktől.
  </li>
</ul>

### Graph rewire
<p>
A graph rewiring szimuláció azt mutatja meg, hogy a gráf lokális struktúrájának
fokozatos szétrombolása hogyan befolyásolja a közösségdetektálást és a GCN predikciókat.
</p>
<table>
  <tr>
    <th>Rewiring valószínűség</th>
    <th>Louvain NMI</th>
    <th>Louvain ARI</th>
    <th>GCN NMI</th>
    <th>GCN ARI</th>
  </tr>
  <tr>
    <td>0.0</td>
    <td>0.452</td>
    <td>0.261</td>
    <td>0.504</td>
    <td>0.505</td>
  </tr>
  <tr>
    <td>0.1</td>
    <td>0.352</td>
    <td>0.163</td>
    <td>0.442</td>
    <td>0.446</td>
  </tr>
  <tr>
    <td>0.2</td>
    <td>0.253</td>
    <td>0.117</td>
    <td>0.374</td>
    <td>0.381</td>
  </tr>
  <tr>
    <td>0.4</td>
    <td>0.143</td>
    <td>0.054</td>
    <td>0.260</td>
    <td>0.255</td>
  </tr>
  <tr>
    <td>0.6</td>
    <td>0.059</td>
    <td>0.005</td>
    <td>0.172</td>
    <td>0.155</td>
  </tr>
</table>

<div style="display: flex; justify-content: center;">
<img width="45%"  alt="ari_rewire" src="https://github.com/user-attachments/assets/a3f3e1a9-bfb2-4d61-ba5b-51ed68de214d" />
<img width="45%"  alt="nmi_rewire" src="https://github.com/user-attachments/assets/8410a35d-5bdb-4006-b986-4c980e4b0418" />
</div >
<ul>
  <li>
    <strong>Louvain:</strong><br>
    Már kis rewiring valószínűségnél (p = 0.1) is jelentős teljesítménycsökkenés figyelhető meg.
    A közösségstruktúra gyorsan szétesik, mivel a Louvain erősen függ a lokális sűrűségi mintázatoktól.
    Nagy rewiring esetén (p ≥ 0.4) a módszer gyakorlatilag összeomlik.
  </li>
  <li>
    <strong>GCN:</strong><br>
    A GCN teljesítménye fokozatosan csökken, de minden rewiring szinten
    jelentősen jobb marad, mint a Louvain eredménye.
  </li>
</ul>

### Rewire with Triadic Closure
<table>
  <tr>
    <th>Rewiring valószínűség</th>
    <th>Louvain NMI</th>
    <th>Louvain ARI</th>
    <th>GCN NMI</th>
    <th>GCN ARI</th>
  </tr>
  <tr>
    <td>0.0</td>
    <td>0.446</td>
    <td>0.227</td>
    <td>0.505</td>
    <td>0.507</td>
  </tr>
  <tr>
    <td>0.1</td>
    <td>0.436</td>
    <td>0.253</td>
    <td>0.500</td>
    <td>0.500</td>
  </tr>
  <tr>
    <td>0.2</td>
    <td>0.427</td>
    <td>0.224</td>
    <td>0.487</td>
    <td>0.485</td>
  </tr>
  <tr>
    <td>0.4</td>
    <td>0.410</td>
    <td>0.199</td>
    <td>0.478</td>
    <td>0.485</td>
  </tr>
  <tr>
    <td>0.6</td>
    <td>0.387</td>
    <td>0.159</td>
    <td>0.461</td>
    <td>0.469</td>
  </tr>
</table>
<div style="display: flex; justify-content: center;">
<img width="45%" alt="ari_rewire_triadic" src="https://github.com/user-attachments/assets/3b1efc40-d80e-459f-93a4-bbe3efafc5a8" />
<img width="45%" alt="nmi_rewire_triadic" src="https://github.com/user-attachments/assets/13c0deed-6b83-4cf1-be8a-bf9fc21827e5" />

</div >

### Targeted attacks
<p>
  A targeted attack szimuláció célja annak vizsgálata, hogy mi történik, ha nem véletlenszerűen, hanem strukturálisan legfontosabb csomópontokat támadjuk meg.
</p>
<table>
  <tr>
    <th>Attack arány</th>
    <th>k csomópont</th>
    <th>Louvain Hub NMI</th>
    <th>GCN Hub NMI</th>
    <th>Louvain Bet NMI</th>
    <th>GCN Bet NMI</th>
  </tr>
  <tr>
    <td>1%</td>
    <td>27</td>
    <td>0.410</td>
    <td>0.490</td>
    <td>0.421</td>
    <td>0.491</td>
  </tr>
  <tr>
    <td>3%</td>
    <td>81</td>
    <td>0.384</td>
    <td>0.489</td>
    <td>0.409</td>
    <td>0.494</td>
  </tr>
  <tr>
    <td>5%</td>
    <td>135</td>
    <td>0.377</td>
    <td>0.484</td>
    <td>0.403</td>
    <td>0.499</td>
  </tr>
</table>

<div style="display: flex; justify-content: center;">
<img width="45%"  alt="ari_targeted_attacks" src="https://github.com/user-attachments/assets/43b71b20-e63f-4680-9a93-861671715c0f" />
<img width="45%" alt="nmi_targeted_attacks" src="https://github.com/user-attachments/assets/dc105ac2-e888-48d7-8252-3b3410f1d26f" />
</div >

<p>
A célzott támadások súlyosabb strukturális károsodást okoznak,
mint a véletlenszerű edge dropout vagy a rewiring.
Ennek ellenére a GCN robusztusabb marad, mint a Louvain módszer.
</p>

<ul>
  <li>
    <strong>Hub támadás:</strong><br>
    A legnagyobb fokszámú csomópontok éleinek eltávolítása gyorsan rontja
    a Louvain közösségdetektálás teljesítményét.
    Már 1–3% támadási arány esetén is jelentős ARI/NMI a csökkenés.
  </li>

  <li>
    <strong>Betweenness támadás:</strong><br>
    A közvetítő szerepű csomópontok eltávolítása különösen fájdalmas,
    mivel ezek a csomópontok a közösségek közti információáramlás kulcspontjai.
    A Louvain teljesítménye tovább romlik, míg a GCN alig veszít pontosságából.
  </li>
</ul>







