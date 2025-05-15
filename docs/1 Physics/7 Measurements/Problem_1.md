# Problem 1
## Measuring Earth's Gravitational Acceleration using a Pendulum

## Step-by-Step Solution

---

### 1. Data Overview

- **Pendulum Length**:  
$$ L = 1.25 \ \text{m} $$

- **Ruler Resolution**:  
$$ \text{Ruler Resolution} = 0.01 \ \text{m} $$

- **Uncertainty in Length**:
$$
\Delta L = \frac{0.01}{2} = 0.005 \ \text{m}
$$

---

### 2. Time Analysis

- **Mean time for 10 oscillations**:
$$
\overline{T_{10}} = 15.265 \ \text{s}
$$

- **Standard deviation**:
$$
\sigma_T = 0.0303 \ \text{s}
$$

- **Uncertainty in mean time**:
$$
\Delta T_{10} = \frac{\sigma_T}{\sqrt{n}} = \frac{0.0303}{\sqrt{10}} = 0.0096 \ \text{s}
$$

---

### 3. Period Calculation

- **Period of 1 oscillation**:
$$
T = \frac{\overline{T_{10}}}{10} = 1.5265 \ \text{s}
$$

- **Uncertainty in period**:
$$
\Delta T = \frac{\Delta T_{10}}{10} = 0.00096 \ \text{s}
$$

---

### 4. Calculate Gravitational Acceleration

Using the pendulum formula:
$$
g = \frac{4\pi^2 L}{T^2}
$$

Plug in values:
$$
g = \frac{4\pi^2 \cdot 1.25}{(1.5265)^2}
$$

Let's calculate this:
$$
g = \frac{4 \times 9.8696 \times 1.25}{(1.5265)^2} = \frac{49.348}{2.334} \approx 9.82 \ \text{m/s}^2
$$

So, the value of \( g \) is approximately \( 9.82 \ \text{m/s}^2 \), which matches the standard value for gravitational acceleration on Earth.

---

### 5. Uncertainty Propagation

To calculate the uncertainty in \( g \), we use the following formula:
$$
\Delta g = g \cdot \sqrt{\left( \frac{\Delta L}{L} \right)^2 + \left( 2 \cdot \frac{\Delta T}{T} \right)^2 }
$$

Substitute the known values:

- \( \frac{\Delta L}{L} = \frac{0.005}{1.25} = 0.004 \)
- \( \frac{\Delta T}{T} = \frac{0.00096}{1.5265} \approx 0.000629 \)

Now, calculate \( \Delta g \):

$$
\Delta g = 9.82 \cdot \sqrt{(0.004)^2 + (2 \times 0.000629)^2}
$$

$$
\Delta g = 9.82 \cdot \sqrt{0.000016 + 0.000792}
$$

$$
\Delta g = 9.82 \cdot \sqrt{0.000808} \approx 9.82 \cdot 0.0284 \approx 0.28 \ \text{m/s}^2
$$

Thus, the uncertainty in \( g \) is approximately \( 0.28 \ \text{m/s}^2 \).

---

### Final Results Table

| Quantity            | Value         |
|---------------------|---------------|
| $L$                 | 1.25 m        |
| $\Delta L$          | 0.005 m       |
| $\overline{T_{10}}$ | 15.265 s      |
| $\sigma_T$          | 0.0303 s      |
| $\Delta T_{10}$     | 0.0096 s      |
| $T$                 | 1.5265 s      |
| $\Delta T$          | 0.00096 s     |
| $g$                 | 9.82 m/s²     |
| $\Delta g$          | 0.28 m/s²     |

---

To understand how much your time measurements vary from the average, you calculate the **standard deviation**. Here's a step-by-step guide:

---

### Step-by-Step Formula

Given a set of time measurements:
$$
T_{10} = [t_1, t_2, ..., t_n]
$$

#### 1. Calculate the Mean:
$$
\overline{T_{10}} = \frac{1}{n} \sum_{i=1}^{n} t_i
$$

#### 2. Find the Deviation for Each Time:
$$
\text{Deviation}_i = t_i - \overline{T_{10}}
$$

#### 3. Square Each Deviation:
$$
(\text{Deviation}_i)^2
$$

#### 4. Sum the Squared Deviations:
$$
\sum_{i=1}^{n} (t_i - \overline{T_{10}})^2
$$

#### 5. Divide by Degrees of Freedom (n - 1):
$$
\frac{1}{n - 1} \sum_{i=1}^{n} (t_i - \overline{T_{10}})^2
$$

#### 6. Take the Square Root:
$$
\sigma_T = \sqrt{\frac{1}{n - 1} \sum_{i=1}^{n} (t_i - \overline{T_{10}})^2}
$$

### Example

Given time measurements:
$$
T₁₀ = [15.22, 15.31, 15.26, 15.28, 15.30, 15.24, 15.25, 15.29, 15.23, 15.27]
$$

- **Mean**:
  $$
  \overline{T_{10}} = 15.265 \ \text{s}
  $$

- **Sample Deviation**:
  - $15.22 - 15.265 = -0.045$
  - $15.31 - 15.265 = 0.045$
  - etc.

- **Square of deviations (sample)**:
  - $(-0.045)^2 = 0.002025$
  - $(0.045)^2 = 0.002025$
  - etc.

- **Sum of squares**:  
  $$
  \sum (\text{Deviation}^2) = 0.00825
  $$

- **Divide by degrees of freedom**:  
  $$
  \frac{0.00825}{9} = 0.000916
  $$

- **Standard deviation**:  
  $$
  \sigma_T = \sqrt{0.000916} \approx 0.0303 \ \text{s}
  $$

---

This standard deviation tells you the **spread** or **variability** in your timing data. A smaller value means your measurements are more consistent!

### Notes

- The standard value of $g$ is approximately **9.81 m/s²**.
- The discrepancy may arise due to:
  - Timing errors.
  - Air resistance.
  - Angle of release exceeding small-angle approximation.
  - Measurement resolution limitations.
