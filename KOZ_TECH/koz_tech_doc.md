# Közösségi és technológiai hálózatok - Dáné Bence
asd
asd
asd

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
asd
asd
<div style="display: flex; justify-content: center;">
  <img alt="original_feature_hulls" src="https://github.com/user-attachments/assets/a40592ad-6917-42d1-85ad-3f5f98fbb7d8" width="45%"/>
  <img alt="gnn_embedding_hulls" src="https://github.com/user-attachments/assets/778dc5e0-afd2-4c37-b544-a02810016333"  width="45%"/>
</div>
asd
asd
asd
<details>
  <summary><strong>1. k-NN Graph – Embedding Communities</strong></summary>
  <img src="https://github.com/user-attachments/assets/7530e5bf-3ab0-46c1-a6bf-837b518b6087" width="75%">
</details>

<details>
  <summary><strong>2. k-NN Graph – True Labels</strong></summary>
  <img src="https://github.com/user-attachments/assets/14b8d691-8d7d-41a4-8c14-e4cd4b8cbd46" width="75%">
</details>

<details>
  <summary><strong>3. k-NN Graph – Original Graph Communities</strong></summary>
  <img src="https://github.com/user-attachments/assets/65f7125a-422d-415c-a20d-2d56b693c0c8" width="75%">
</details>

<details>
  <summary><strong>4. k-NN Graph – Nodes That Switched Community</strong></summary>
  <img src="https://github.com/user-attachments/assets/b2950fbb-ba90-4870-8812-ec7c5465f466" width="75%">
</details>



