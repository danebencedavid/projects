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



