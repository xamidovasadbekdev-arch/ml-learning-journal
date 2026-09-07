# 🚀 Linear Regression & Gradient Descent from Scratch (CCPP)

Ushbu loyiha **Combined Cycle Power Plant (CCPP)** datasetida Chiziqli Regressiya hamda Gradient Descent (GD) optimizatsiya algoritmining 3 xil rejimini (**Batch GD**, **SGD** va **Mini-batch GD**) Python va NumPy yordamida **noldan (from scratch)** amalga oshirish va tahlil qilishga bag'ishlangan.

---

## 📌 Loyiha Haqida

* **Dataset:** Combined Cycle Power Plant (UCI ML Repository) — 9,568 ta namuna
* **Target (`PE`):** Soatlik elektr energiyasi ishlab chiqarish quvvati (MW)
* **Features:** Atrof harorati (`AT`), vakuum (`V`), atmosfera bosimi (`AP`), nisbiy namlik (`RH`)
* **Maqsad:** Tayyor optimizatorlarsiz, matritsa va vektorlar darajasida GD ishlatib, optimal $\theta$ parametrlarni topish hamda model xatti-harakatlarini tahlil qilish.

---

## 🛠️ Texnologiyalar

* **Python 3.x**
* **NumPy** — Vektorlashgan matematik va matritsa amallari
* **Pandas** — Datasetni qayta ishlash va tahlil qilish
* **Matplotlib / Seaborn** — Loss egri chiziqlari hamda Contour vizualizatsiyasi
* **Scikit-Learn** — Data splitting, StandardScaler va Baseline modellar bilan solishtirish

---

## 🔍 O'tkazilgan Inspeksiyalar va Xulosalar

### 1. Vectorization & Gradient Calculation
* **Gradient Formulasi:** $\nabla_{\theta} J(\theta) = \frac{2}{m} X_b^T (X_b \theta - y)$
* Bias ($b$) va og'irliklarni ($w$) alohida saqlamasdan, $X$ matritsasiga $1$-lar ustuni qo'shildi ($X_b = [1, X]$). Bu barcha amallarni yagona $\theta$ vektori bilan tez va samarali bajarishga imkon berdi.

### 2. Learning Rate & Divergensiya
* `learning_rate >= 0.5` qiymatlarda loss bir zumda portlab (`inf` / `nan`), GD kosa devorlariga urilib uzoqlashib ketdi (divergensiya).
* Scaled ma'lumotlarda eng barqaror va tez konvergensiya beruvchi qiymat **`lr = 0.01`** deb topildi.

### 3. Optimizer Rejimlarining Solishtirmasi
* **Batch GD (`batch_size=None`):** Trayektoriyasi juda silliq va aniq, ammo har bir epochta butun ma'lumotni hisoblagani uchun sekin.
* **SGD (`batch_size=1`):** Har bir qadam tez, lekin yakka namunalar shovqini tufayli loss trayektoriyasi "zigzag" tebrandi.
* **Mini-batch GD (`batch_size=32`):** **Eng optimal rejim.** Hardware (CPU SIMD) parallellashtirishidan to'liq foydalanadi hamda vaqt va aniqlik bo'yicha eng yaxshi natijani ko'rsatdi.

### 4. Baseline Modellar Bilan Solishtirish
* Custom GD modelingiz natijalari analitik **Normal Equation** ($(X^T X)^{-1} X^T y$) va Scikit-Learn'ning `SGDRegressor`i bilan solishtirildi.
* Epochlar soni oshirilishi bilan Custom GD parametrlari analitik yechimga to'liq yaqinlashdi ($R^2 \approx 0.93$).

### 5. Feature Scaling Ta'siri
* Scale qilinmagan $X$ ma'lumotlarida loss funksiyasi fazosi o'ta cho'zilgan **ellips** hosil qilgani sababli standart `lr`da model divergensiyaga uchradi.
* `StandardScaler` qo'llanilganda kosa shakli sferiklashdi va GD qadamlari bir necha barobar tezlashdi.

### 6. Contour Vizualizatsiyasi (GD Yo'li)
* $(θ_0, θ_1)$ parametrlar to'rida MSE contour chizilib, Batch GD ning to'g'ridan-to'g'ri markazga boruvchi silliq yo'li hamda SGD ning markaz atrofidagi tebranuvchi zigzag trayektoriyasi yaqqol isbotlandi.

### 7. Feature Engineering
* `AT` va `PE` o'rtasidagi biroz chiziqsiz (egri) bog'liqlikni hisobga olib, `AT**2` (kvadratik atribut) qo'shildi va modelning $R^2$ ko'rsatkichida o'sishga erishildi.
