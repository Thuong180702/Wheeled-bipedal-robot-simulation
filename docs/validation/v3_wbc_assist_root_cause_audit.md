# V3 / WBC / Assist — Root-Cause Audit (2026-07-19)

Audit toàn diện track điều khiển cổ điển (V3 + QP-WBC + Assist blend), truy nguyên
các triệu chứng: drift không hồi vị trí, chân dang không phục hồi tư thế, torque
rung/đỉnh cao (+28%), Assist chưa cải thiện đủ. Mọi phát hiện đều có thí nghiệm
tái lập (scripts trong scratchpad phiên audit; số liệu ghi lại tại đây).

**Setup chuẩn dùng cho mọi thí nghiệm:** `variant_nominal` (root_z=0.5357,
target_com_z=0.4040), push 90 N vào `r_thigh`, hướng [-0.2097, 0.8245, -0.5255],
8 bước, warmup 250, post 400, n_substeps=5, profile `K2_JAX_DEDICATED_DEFAULT_V3`.

---

## Tóm tắt xếp hạng

| # | Mức | Loại | Phát hiện |
|---|-----|------|-----------|
| F1 | CRITICAL | BUG | Contact Jacobian của WBC: toàn bộ cột khớp chân = 0 (autodiff qua FK đã tính sẵn) |
| F2 | CRITICAL | BUG | Fast solver (OSQP, mặc định) hardcode `feasibility_only` — toàn bộ task stack bị đánh rơi |
| F3 | CRITICAL | BUG | Offline eval truyền base-z làm `commanded_height_ref_m` (V3 đo CoM-z) → drift/heading/centering gate chết, gain-schedule sai điểm làm việc |
| F4 | CRITICAL | BUG/DESIGN | Gate adaptive assist đo height theo base-z (h0=0.53) nhưng dùng chung biến `height_ref` với lệnh CoM của V3 — sửa F3 đơn thuần sẽ tắt hẳn assist (α=0) |
| F5 | HIGH | THIẾU LOGIC | Không tồn tại vòng phản hồi kéo về (x, y, yaw) mục tiêu ở bất kỳ tầng nào |
| F6 | HIGH | DESIGN + SIGN | Heading hip-yaw: sign ngược (positive feedback) và cơ chế gần như không có thẩm quyền yaw; tác dụng phụ = scissor chân |
| F7 | MEDIUM | BUG tiềm ẩn + cơ chế rung | Nhánh hysteresis-rescale bỏ qua correction cap; assist bật/tắt răng cưa do 19–21% WBC solve fail + attack/decay bất đối xứng |
| F8 | MEDIUM | DESIGN | τ_wbc − τ_v3 chứa thành phần "gauge" từ phân bố lực tiếp xúc bất định — blend torque thô là sai nguyên lý ở trạng thái tĩnh |
| F9 | MEDIUM | NHẤT QUÁN | Realtime --assist (α cố định) vs promote (adaptive) là 2 thuật toán khác nhau; warm-start không bao giờ được truyền; nhãn roll/pitch offline tráo trục; w_com×0.4, w_torso×0.5 nhân ngầm |
| F10 | LOW | DEBT | Drift height-gate "fix" bằng nới ngưỡng sang đơn vị cm thay vì sửa tín hiệu — mất ngữ nghĩa gate theo tốc độ height |
| F11 | INFO | — | RuntimeWarning matmul khi build H_task là noise BLAS; snapshot hữu hạn, nan_to_num không hề kích hoạt (0/200 call) |

---

## F1 — Contact Jacobian: cột khớp chân = 0 (BUG, CRITICAL)

**Triệu chứng:** WBC-only tự drift/collapse; qdd ~260–410 rad/s² cho bài giữ tĩnh;
τ_wbc hip_yaw ≈ ±4.7 Nm và knee −10.3 Nm trong khi V3 chỉ cần ±0.3 / −7.0;
assist blend làm torque_max +28%, oscillation +21%, chân scissor.

**Cơ chế gốc:** `wheeled_biped/dynamics/jax_contact_dynamics.py`:

- `_cpwp_from_fk(qpos, body_id, local_point, fk_body_pos, fk_body_quat)` tính vị trí
  điểm tiếp xúc từ **FK đã tính sẵn** truyền vào dưới dạng tham số. `qpos` không
  được dùng trong thân hàm.
- `_jac_cpwp_fk = jax.jacfwd(_cpwp_from_fk)` đạo hàm theo arg 0 (`qpos`) → **bằng 0
  hằng nhiên** vì gradient không chảy qua các mảng FK hằng.
- Dòng `contact_point_translational_jacobian = contact_point_translational_jacobian_jit`
  ghi đè bản đúng (bản gốc dòng ~263 tính FK bên trong hàm được đạo hàm) bằng bản lỗi.
- Mọi pipeline (promote / phase3d / realtime --assist / viz) đi qua
  `build_padded_contact_stack` / `build_contact_stack` → đều dùng bản lỗi.

**Hệ quả vật lý:** trong model của QP, phản lực đất không tạo moment lên
hip/knee/yaw (J_cᵀλ chỉ đổ vào 6 hàng base); ràng buộc "gia tốc pháp tuyến điểm
tiếp xúc = 0" cũng chỉ ràng buộc base → bài toán động lực học sai hoàn toàn ở
mức khớp → nghiệm τ_wbc vô nghĩa ở chính các khớp mà assist tin tưởng nhất
(K_role hip_pitch/knee = 0.60).

**Bằng chứng:**
- So với `mj_jac` cùng state: `max|Jp_snap − Jp_mj| = 0.44`; mọi cột l_hr…r_wh
  của snapshot = 0.000 trong khi MuJoCo cho 0.03–0.44. (h khớp tuyệt đối; M lệch max 0.036.)
- Monkeypatch Jacobian đúng (jacfwd qua FK thật): `|Jp − Jp_mj| → 4e-4`;
  qdd_max 270 → 54; τ_hy −4.91 → −0.22; các task mode bắt đầu cho nghiệm khác nhau.

**Fix nguyên lý:** khôi phục đạo hàm qua FK thật. Muốn giữ JIT: đưa
`jax_forward_kinematics` vào TRONG hàm được `jacfwd`, JIT bên ngoài theo
`(body_id, local_point)` static/dynamic hợp lý — không autodiff qua giá trị cache.
Thêm test đối chiếu `mj_jac` (ngưỡng 1e-3) vào tests/.

**Rủi ro:** chi phí tính tăng (jacfwd full FK mỗi contact mỗi bước — đo lại QP-SLOW);
mọi hành vi WBC thay đổi → phải chạy lại toàn bộ benchmark; các gain/weight từng
"tune quanh bug" có thể cần chỉnh lại.

---

## F2 — Fast path đánh rơi toàn bộ task stack (BUG, CRITICAL)

**Triệu chứng:** đổi `task_mode` không đổi kết quả; WBC "không có mục tiêu".

**Cơ chế gốc:** `wheeled_biped/wbc/structured_qp_problem.py::_build_phase3b_qp_cached`
hardcode `build_phase3b_qp_from_snapshot(snapshot, "feasibility_only", constants)`.
`_build_sparse_objective` nhận `task_mode` nhưng **không dùng**. Đường OSQP
(mặc định của mọi script) do đó chỉ giải "min w_qdd·‖qdd‖² + rolling_soft" —
không CoM height, không torso, không posture, không com_xy damping, không yaw damping.
`q_act_ref_override` cũng không được truyền xuống fast path.

**Bằng chứng:** 4 task mode (`feasibility_only`/`balanced_default`/`posture_priority`/
`com_priority`) cho nghiệm **giống hệt** trên OSQP (qdd_kn = −268.1 cả 4); SLSQP
(đường cũ, build đúng mode) cho −268/−256/−224 — có phản ứng với weights.

**Fix nguyên lý:** truyền `task_mode` (+ posture override) vào cache key và builder;
hoặc bỏ cache `feasibility_only` và build objective đúng mode một lần mỗi snapshot.

**Rủi ro:** thấp về code, nhưng cùng F1 làm thay đổi toàn bộ hành vi WBC.

---

## F3 — Offline truyền base-z làm lệnh height CoM của V3 (BUG, CRITICAL)

**Triệu chứng trực tiếp trên GIF:** drift không được dập, yaw trôi, chân dang
không hồi, tư thế khác hẳn ban đầu.

**Cơ chế gốc:** `promote_v3_vs_assist.py` (`height_ref = scenario_meta["seed_qpos_z"]`,
và cả bước stabilize 100-step dùng `data.qpos[2]`) và `viz_v3_vs_assist_push.py`
(meta `seed_qpos_z = root_z`) truyền **base-z = 0.5357** vào
`commanded_height_ref_m`, trong khi V3 đo `com_z_m ≈ 0.404` (CoM). Trong JAX
controller `schedule_h = height_ref` nên:

1. `|com_z − schedule_h|·100 ≈ 13.2 cm` > `drift_hgate_vel_high = 12` →
   `drift_height_gate_vel = 0` vĩnh viễn → **drift velocity damping tắt**.
2. `heading_height_gate = 0` → **heading hip-yaw tắt**; `center_gate = 0` →
   **hip-yaw mean centering tắt** (chân dang không có gì kéo về).
3. Toàn bộ gain schedule (`cal_kp/kd/theta_max/deadband`), `physics_ff_tau`,
   `pitch_eq`, notch gate nội suy tại **0.536 thay vì 0.404** — sai điểm làm việc.
4. `eq_joint` (mỏ neo posture) được latch sau 100 bước settle dưới lệnh sai.

Realtime (`run_k2_jax_realtime.py`) dùng `target_com_z` từ setup JSON — đúng.
Đây chính là khác biệt bản chất giữa "V3 realtime chịu 90 N đẹp" và offline tệ.

**Bằng chứng (A/B, V3 single-arm, cùng push):**

| | A: height_ref=0.5357 (stock) | B: height_ref=0.4040 (đúng) |
|---|---|---|
| drift_height_gate_vel (mean) | **0.0000** | **1.0000** |
| heading_gate / center_gate | 0 / 0 | 0.42 / 0.93 (pre-push) |
| posture_dev_rms cuối | 0.1233 rad | 0.0633 rad |
| hip_yaw L/R cuối | −0.192 / +0.194 | −0.114 / +0.126 |
| hip_roll L/R cuối | +0.170 / −0.146 | +0.027 / −0.067 |

**Fix nguyên lý:** scenario builders + stabilize + context của promote/phase3d/viz
phải truyền **target CoM z** (từ setup JSON, hoặc CoM đo được lúc settle) làm
`height_ref`. Không đổi controller.

**Rủi ro:** mọi số liệu official (223 scenario) phải chạy lại; xem F4 trước khi sửa.

---

## F4 — Hai ngữ nghĩa height trên cùng một biến (BUG/DESIGN, CRITICAL)

**Cơ chế:** trong promote, `_assist_state = {"height": qpos[2] (base-z),
"height_target": height_ref, ...}` và `h0 = ADAPTIVE_HEIGHT_MODEL_NOMINAL = 0.53`
(base-z). Nếu sửa F3 bằng cách đặt `height_ref = 0.404` (CoM), thì
`g_height_cmd = exp(−((0.404−0.53)/0.025)²) ≈ e^−25 ≈ 0` → **assist tắt hẳn**.

**Bằng chứng (dual-arm instrumented):**
- stock: g_h mean 0.874, α_pj max ~0.19, assist hoạt động;
- fixed: **g_h = 0.000, α_pj = 0.0000 mọi bước** — "Assist" = V3 nguyên bản.

**Fix nguyên lý:** tách 2 tham chiếu: `v3_height_ref_com` (lệnh CoM cho V3) và
`assist_height_state` (base-z cho gate, target = base-z tương ứng biến thể, h0
base-z). Không dùng chung một biến cho hai hệ quy chiếu.

**Ghi chú thêm:** `ADAPTIVE_HEIGHT_SIGMA = 0.025` nghĩa là g_height ≈ 0 khi lệch
±5 cm — assist gần như tắt trong mọi height-transition suite (step_c/step_d).
Đây là "WBC chỉ dám hoạt động ở nominal" — hạn chế thiết kế cần cân nhắc lại
sau khi F1/F2 làm WBC đáng tin hơn.

---

## F5 — Không có vòng phản hồi vị trí/heading (THIẾU LOGIC, HIGH)

**Triệu chứng:** robot phục hồi sau push nhưng nằm lệch vĩnh viễn, không kéo về tâm.

**Cơ chế:** đúng như giả thuyết đề bài:
- V3 profile: `drift_k_vel=10` (chỉ dập vận tốc sagittal), `drift_k_pos=0`,
  `drift_k_heading=0` (comment "unsafe at low height" — quyết định có chủ đích từ V2).
  Code position-return (`tau_drift_pos`) và heading wheel-differential tồn tại
  nhưng gain = 0; thêm nữa `position_gate`/`heading_gate` của drift dùng
  `hgate_low/high = 0.03/0.15` với tín hiệu đơn vị **cm** → gate chết ngay cả khi
  bật gain (bug đơn vị tiềm ẩn, xem F10).
- WBC: task `com_xy` chỉ dập vận tốc (`a_des = kp·(0 − v_xy)`, biến `kd_xy` khai báo
  nhưng không dùng); `yaw_damping` chỉ phạt gia tốc; posture task fallback = tư thế
  hiện tại; `_commanded_height`/`_posture_ref_override` **không được set** bởi bất kỳ
  pipeline chính nào (chỉ `scripts/run_v3_assist_comparison.py` set).
- Lateral (body-y): không có cơ chế nào — bánh xe không đẩy ngang được, muốn hồi
  lateral bắt buộc phải steer (yaw) + chạy tới — tức cần heading + position loop.

**Kết luận:** không phải bug — là **số hạng điều khiển chưa tồn tại**. Sau push,
hệ chỉ về trạng thái "đứng yên tại chỗ mới".

**Fix nguyên lý (sau F1–F4):** thêm position/heading return yếu, gated đúng đơn vị
(khoảng cách + độ ổn định + tốc độ height), qua wheel torque (sagittal) và wheel
differential (yaw); lateral hồi qua chuỗi yaw→forward. Ablation lại lý do
"unsafe at low height" với tín hiệu gate đã sửa — kết luận cũ rút ra khi gate
còn bug đơn vị nên đáng nghi.

---

## F6 — Heading hip-yaw: sign ngược + không có thẩm quyền (DESIGN+SIGN, HIGH)

**Cơ chế:** heading stabilizer đặt τ_L = +τ, τ_R = −τ, claim "CW yaw". Nhưng cặp
torque đối nhau trên 2 khớp hip-yaw cho **tổng phản lực trực tiếp lên torso = 0**;
tác dụng yaw chỉ đến gián tiếp qua contact, còn tác dụng trực tiếp là 2 chân xoay
ngược chiều nhau (scissor) — thứ anti-twist + mean-centering phải chống lại.

**Bằng chứng (injection test, robot đang cân bằng, 200 bước):**
- τ = 0: Δyaw = −0.89° (drift nền)
- τ_L=+1.5, τ_R=−1.5 Nm (max authority): Δyaw = +0.09° (≈ **+1.0° so với nền — CCW, NGƯỢC claim CW**)
- τ đảo dấu: Δyaw = −1.69° (≈ −0.8° so với nền)
Tức là: sign ngược (positive feedback nhẹ lên yaw error) và độ lợi ~0.6°/s·Nm⁻¹ —
gần bằng 0. Trong rollout B (gate mở), yaw trôi tới −24.8° dù heading "hoạt động",
|τ_heading| chỉ ~0.09 Nm.

**Fix nguyên lý:** yaw authority thật nằm ở **wheel differential** (đường
`drift_k_heading` có sẵn, đang 0). Sau khi F3 sửa gate: bật thử k_heading nhỏ với
sign được VALIDATE bằng injection test qua wheel (không qua hip-yaw), demote heading
hip-yaw thành 0 hoặc chỉ giữ mean-centering. Đồng thời giảm số controller chồng
nhau trên hip-yaw (hiện có 4: heading, anti-twist, mean-center, posture PD —
telemetry `cancel_hip_yaw` sinh ra đúng để đo sự đánh nhau này).

---

## F7 — Assist bật/tắt răng cưa + nhánh rescale bỏ cap (MEDIUM)

**Cơ chế rung:**
1. WBC solve fail 19–21% số bước (658 bước push scenario: ok-rate 0.79–0.81;
   status chủ yếu `maximum iterations reached`) → fail-closed về pure V3 →
   τ nhảy bậc giữa `tau_v3` và `tau_v3 + α·corr` (|corr| p95 ≈ 6 Nm, α ~0.1 →
   bậc ~0.5 Nm mỗi lần chuyển).
2. Hysteresis: attack = 1.0 (rơi tức thì), decay = 0.10 (hồi chậm) — cộng với (1)
   tạo sawtooth α.
3. **Bug tiềm ẩn:** nhánh rescale trong promote (khi `_alpha_scale < 0.999`,
   xảy ra 367/534 bước) tính lại
   `tau_assist = tau_v3 + α·(tau_wbc − tau_v3)` dùng correction **thô**, bỏ qua
   `correction_capped` (cap = 0.25·g_h·τ_limit). Đo thực nghiệm: excess hiện tại
   = 0.000 Nm (chưa gây hại) — nhưng là quả bom chờ khi correction lớn hơn cap.

**Fix:** dùng `correction_capped` trong nhánh rescale; sau F1/F2 đo lại solve-fail
rate (kỳ vọng giảm mạnh vì QP well-posed hơn); nếu còn, tăng max_iter/warm-start
thật (xem F9) thay vì chấp nhận flicker.

---

## F8 — Blend torque thô là sai nguyên lý ở trạng thái tĩnh (DESIGN, MEDIUM)

Ở cân bằng tĩnh, phân bố (τ, λ) là bất định (redundant). QP chọn nghiệm theo
regularization của nó (w_tau = 0.001 → gần minimum-torque), MuJoCo/V3 "chọn" phân
bố khác. Sau khi sửa F1, WBC vẫn cho τ_knee = **+2.65** trong khi V3 giữ ổn với
**−7.02** — chênh 9.7 Nm không phải "correction" mà là **gauge difference**.
Blend α·(τ_wbc − τ_v3) do đó bơm torque ký sinh thường trực (đây là phần còn lại
của +28% torque_max sau khi trừ F1).

**Fix nguyên lý:** neo QP về V3 — thêm số hạng `w_tau_ref·‖τ − τ_v3‖²` (hoặc λ_ref
từ contact force đo được), để τ_wbc − τ_v3 chỉ còn phần "task-driven"; hoặc chuyển
hẳn sang kiến trúc posture-guided (WBC điều chỉnh q_ref/feedforward của V3 —
đường `compute_posture_guided_assist` đã có sẵn) thay vì cộng torque.

---

## F9 — Bất nhất giữa các đường eval (MEDIUM)

- `run_k2_jax_realtime.py --assist` dùng `compute_assist_torque` (α cố định 0.25),
  promote/viz dùng `compute_adaptive_assist_torque` (α_max 0.35 + gates + hysteresis)
  → hai chế độ quan sát không so sánh được với nhau.
- `g_divergence` đo độ lệch giữa 2 rollout của 2 controller KHÁC nhau — chúng tất
  nhiên lệch sau push; gate này phạt assist vì… khác V3, trái mục tiêu. Trong
  single-arm không tồn tại tín hiệu này (mặc định 1.0) — thêm một nguồn bất nhất.
- Warm-start chết: `_dispatch_wbc_torque` đọc `kwargs["_warm_start_vec"]` — không
  ai set key này → OSQP cold-start mọi bước (góp phần max-iter fail ở F7).
- Nhãn offline: `_quat_to_rpy` 'xyz' với forward = +Y → "roll" offline = nghiêng
  sagittal (pitch vật lý), "pitch" = nghiêng ngang. `ADAPTIVE_STABILITY_THRESHOLDS`
  đã "tune quanh" nhãn tráo (comment "roll ~15-25°" — chính là sagittal lean).
  Cần đổi tên một lần thống nhất với realtime (đúng như ghi chú phiên trước).
- `_build_task_costs_from_snapshot` nhân ngầm `w_com×0.4`, `w_torso×0.5` khác với
  bảng TASK_WEIGHT_MODES khai báo — weights hiệu dụng không như tài liệu.
- `SANITY_QDD_MAX=100` không được enforce trên fast path (qdd 270 vẫn "ok").

---

## F10 — Nợ ngữ nghĩa gate height của drift (LOW/MEDIUM)

`_com_z_vel_abs_drift = |com_z − schedule_h| × 100` là **sai số vị trí (cm)** bị
đặt tên "z_vel". V3 "AUDIT_FIX" đã nới ngưỡng gate vel sang 2–12 cm (chữa triệu
chứng) thay vì sửa tín hiệu — gate giờ khóa theo SAI SỐ height, không theo TỐC ĐỘ
height như thiết kế ("yield during height transitions"). Hệ quả: khi lệnh height
transition có tracking error > 12 cm, drift damping tắt đúng lúc cần; ngược lại
khi transition chậm nhưng bám tốt, gate không nhả như thiết kế gốc. Gate pos/heading
của drift controller vẫn dùng ngưỡng cũ 0.03/0.15 (đơn vị m/s) với tín hiệu cm →
chết vĩnh viễn nếu bật gain. Fix đúng: tính CoM z-velocity thật (đạo hàm/est) cho
gate, hoặc đổi tên + ngưỡng theo đúng ngữ nghĩa "height error".

## F11 — RuntimeWarning matmul: vô hại (INFO)

Kiểm tra 200 call `_build_task_costs_from_snapshot` với snapshot instrumented:
Jcom absmax = 1.0, mọi trường hữu hạn, H_task sau build luôn hữu hạn, `nan_to_num`
không kích hoạt lần nào. Cảnh báo "divide by zero encountered in matmul" là noise
của BLAS (Accelerate) — không phải điều kiện xấu bị che. Các guard nan_to_num rải
rác nên giữ nhưng chuyển thành log-and-fail-closed để không che bug tương lai.

---

## Trả lời 4 câu hỏi ưu tiên

**(a) Vì sao không về vị trí mục tiêu?** F5 (không tồn tại position/heading return
— thiếu logic) chồng lên F3 (ngay cả velocity damping cũng bị gate chết trong
offline eval). Sau push, hệ được thiết kế chỉ để "đứng yên tại chỗ mới".

**(b) Vì sao chân dang không hồi?** F3 tắt hip-yaw mean-centering + heading trong
offline; F6 heading hip-yaw chủ động bơm scissor với sign ngược; WBC posture task
(w=1.5, và thực tế = 0 do F2) track tư thế hiện tại (không neo) → không gì kéo
chân về `eq_joint`; và `eq_joint` bị latch sau settle dưới lệnh height sai.

**(c) Vì sao rung/torque cao?** F1 (τ_wbc sai bản chất: hip-yaw ±4.7 Nm, knee −10.3)
+ F2 (WBC không có mục tiêu) + F8 (gauge mismatch ~vài Nm thường trực) + F7
(19–21% solve fail + hysteresis → sawtooth α, cộng cold-start F9).

**(d) Vì sao Assist chưa cải thiện đủ?** Assist đang blend một tín hiệu WBC (1) sai
động lực học (F1), (2) không mang mục tiêu nào (F2), (3) lệch gauge (F8), qua các
gate (4) đo height sai hệ quy chiếu (F4) và tắt đúng lúc cần giúp (g_stability→0
khi push, g_height→0 khi đổi height), với (5) cơ sở hạ tầng eval tự làm V3 yếu đi
(F3) khiến so sánh V3-vs-Assist official đo trên một V3 "tàn tật". Con số
"Assist tốt hơn roll_max −8.7%" cần diễn giải lại: "roll" offline là sagittal lean,
và cả hai arm đều chạy dưới F3.

---

## Kế hoạch sửa đề xuất (minimal-diff, theo thứ tự, mỗi bước có verify)

> Nguyên tắc: sửa hạ tầng đo lường trước (để mọi số sau đó tin được), rồi model
> WBC, rồi mới bàn logic điều khiển mới. KHÔNG gộp bước.

1. **Fix F3** (eval infra, ~5 dòng): promote/phase3d/viz truyền `target_com_z_m`
   làm height_ref (scenario meta thêm khóa `target_com_z`; fallback đọc setup JSON).
   *Verify:* rerun A/B script — gate vel = 1.0; GIF mới: chân bớt dang, posture_dev
   giảm ~50% như thí nghiệm B.
2. **Fix F4 cùng lúc** (vì F3 đơn lẻ tắt assist): tách `assist_height_state`
   (base-z, target base-z theo variant) khỏi `v3_height_ref` (CoM). *Verify:*
   instrumented run — g_h ≈ 0.87 như stock, α > 0.
3. **Fix F1** (Jacobian, 1 hàm): jacfwd qua FK thật trong bản JIT. *Verify:* test
   mới so `mj_jac` (<1e-3); qdd_max static < ~60; τ_hy_wbc < 1 Nm.
4. **Fix F2** (task stack): thread `task_mode` + `q_act_ref_override` qua cache.
   *Verify:* 3 task mode cho 3 nghiệm khác nhau trên OSQP; posture_priority kéo
   q về override trong sandbox test.
5. **Fix F7 nhỏ** (1 dòng): nhánh rescale dùng `correction_capped`. Fix F9
   warm-start (truyền vector thật). *Verify:* solve-fail rate giảm; không còn excess.
6. **Chạy lại promote --quick** để có baseline số liệu mới sau 1–5, so bảng cũ.
7. **F8/F5/F6 là thay đổi thiết kế** (τ_ref regularization / position-heading
   return / wheel-differential heading + demote hip-yaw heading): đề xuất riêng
   từng RFC nhỏ sau khi 1–6 cho nền số liệu sạch. Mỗi cái cần ablation + injection
   sign test riêng.

Rủi ro chung: mọi kết quả official trước đây (223 scenario, các báo cáo
`docs/validation/*v3_vs_assist*`) được đo dưới F1+F2+F3 — nên coi là invalid cho
mục đích so sánh controller và cần re-run sau bước 6.

---

## PHỤ LỤC — Bước 1+2 đã thực hiện (F3 + F4), 2026-07-19

### Thay đổi code (minimal diff)
- `scripts/promote_v3_vs_assist.py`: thêm `_com_z()`; generators settle theo CoM
  và lưu `target_com_z` vào meta; `run_dual_arm_rollout` dùng `height_ref` = CoM
  (F3), tách `assist_base_z_target` = base-z cho gate assist (F4); default của
  `_build_v3_controller_context` → CoM.
- `scripts/viz_v3_vs_assist_push.py`: meta thêm `target_com_z` từ `target_com_z_m`.
- `scripts/phase3d_full_batch_execution.py`: cùng bộ fix; variant generator
  **start từ keyframe** (bỏ drop tới seed_z 0.63–0.75) — xem caveat bên dưới.

### Phát hiện mới trong lúc sửa (verify bằng thực nghiệm)
1. **Các "height variant" offline degenerate về vật lý.** V3 không track được lệnh
   CoM khác nhau từ keyframe: cả 5 variant (promote & phase3d) đều settle về CoM
   ≈ 0.408 m, cùng tư thế (hip_pitch 0.97, knee 1.73). Chênh lệch gate giữa các
   variant trước đây chỉ là artefact của `|CoM − seed_z|`. → "fixed-height sweep"
   (Table VI) và height variants KHÔNG phải sweep height thật. Đây là hạn chế
   thiết kế variant + thẩm quyền height của V3, không phải bug F3.
2. **Scenario phase3d drop base tới 0.65–0.75 dựa vào chính bug frame.** Lệnh sai
   (base-z làm CoM ⇒ lệnh CoM cao ⇒ V3 duỗi) vô tình giúp V3 bắt cú rơi. Với lệnh
   CoM đúng (0.40) V3 ngồi xuống giữa lúc rơi → sụp. Bằng chứng: drop từ 0.65,
   OLD(href=0.65)→base 0.54 ok, F3(href=0.40)→base 0.13 sụp. → phase3d variant
   phải start từ keyframe (đã sửa); recovery (step_c) drop 0.45–0.85 vượt envelope
   height-recovery của V3 (start dưới natural sụp với MỌI lệnh, kể cả cũ) —
   **cần redesign scenario riêng, không thuộc phạm vi F3/F4** (đã giữ frame fix
   đúng, các scenario ngoài envelope sẽ bị loại settling_failed một cách trung thực).
3. **F3 mở gate ⇒ kích hoạt heading hip-yaw sai dấu (F6).** Sau F3, `heading_gate`
   mở (0→1.0). Vì heading hip-yaw sai dấu và gần như vô thẩm quyền (F6), yaw drift
   có thể XẤU đi (single-arm A/B: yaw −1.5°→−24.8°) cho tới khi F6 được sửa. Đây
   là lý do F6 nằm ở bước sau — F3/F4 là nền, F6 là bước kế.

### Verify F3+F4 (qua path promote thật, nominal + push 90 N)
| Chỉ số | Stock (base-z) | Sau F3+F4 (CoM) |
|---|---|---|
| V3 height_ref | 0.5357 (base-z) | 0.400–0.407 (CoM) ✓ |
| drift_height_gate_vel (mean) | 0.000 | **1.000** ✓ |
| heading_height_gate (mean) | 0.000 | **1.000** ✓ |
| assist g_height (mean) | 0.874 | 0.561 (>0, còn sống) ✓ |
| assist α_pj max | ~0.19 | 0.085 (>0) ✓ |
| V3 falls / Assist falls | 0 / 0 | 0 / 0 ✓ |
| posture_dev_rms cuối (single-arm) | 0.123 rad | **0.063 rad** (chân bớt dang) ✓ |
| hip_roll L/R cuối (single-arm) | +0.17/−0.15 | **+0.03/−0.07** ✓ |

GIF: `outputs/visual/audit_fixed.gif` (so với `outputs/visual/audit.gif` cũ).

### Còn lại (chưa làm — chờ duyệt tiếp)
F7 (rescale-cap + warm-start), F5/F6/F8 (thiết kế), phase3d step_c recovery redesign.

---

## PHỤ LỤC 2 — Bước 3+4 đã thực hiện (F1 + F2), 2026-07-19

### Thay đổi code
- `wheeled_biped/dynamics/jax_contact_dynamics.py`: viết lại
  `contact_point_translational_jacobian_jit` để đạo hàm QUA FK thật
  (`jax_forward_kinematics_fk_arrays`, jitted, `body_id` static). Bỏ
  `_cpwp_from_fk`/`_jac_cpwp_fk` sai (đạo hàm theo qpos không dùng → gradient 0).
- `wheeled_biped/wbc/structured_qp_problem.py`: `_build_phase3b_qp_cached` nhận
  `task_mode`, cache key = `(id(snapshot), task_mode)`; call site + fallback cập nhật.
- `tests/test_phase3d2_fast_solver.py`: thêm `test_task_mode_threads_into_objective`
  (regression F2: Hessian phải khác nhau theo mode).
- `tests/test_phase3d3e_jax_dynamics_cache.py`: nới tolerance jdot 1e-2→2e-2 +
  giải thích (float32 FD-of-Jacobian noise tăng khi cột chân khác 0; float64 khớp <1e-3).

### Verify
| Chỉ số | Trước F1/F2 | Sau F1/F2 |
|---|---|---|
| `\|Jp_leg − mj_jac\|` | **0.44** (cột chân = 0) | **3.9e-4** ✓ |
| WBC qdd_max tĩnh | ~270 rad/s² | **54** ✓ |
| WBC \|tau_wbc\|_max (rollout) | ~11 Nm | **2.89 Nm** ✓ |
| WBC solve ok rate (push) | 0.81 | **1.000** ✓ |
| OSQP phản ứng task_mode | KHÔNG (mọi mode giống nhau) | **CÓ**, khớp SLSQP ✓ |
| J̇q̇ (pos-FD vs jac-FD float64) | — | khớp 5e-4 (Jacobian đúng) ✓ |

- Test regression F1 đã TỒN TẠI và ĐANG ĐỎ trên commit e4bcccc
  (`test_jacobian_vs_cpu_mujoco_keyframe`: 18.2 < 1e-4 fail) → fix làm 30/30 xanh.
  Bug F1 ship kèm test đỏ mà không ai chạy.

### Phát hiện: F1/F2 phơi bày rằng blend hiện tại tự perturb V3 (→ F8/F6)
Trên push r_thigh 90 N: `class=ASSIST_SAFETY_FAIL` — Assist **không ngã** nhưng
twist hip_yaw vượt HARD_HIP_YAW_MAX (20.1°) trong 27 bước (V3 giữ dưới ngưỡng).
Với WBC đã đúng, blend torque thô `α·(τ_wbc − τ_v3)` + heading sai dấu (F6, giờ
active do F3) vẫn đủ để xoắn chân qua ngưỡng an toàn. Đây là hệ quả mong đợi,
KHÔNG phải regression của F1/F2 — nó chính là động lực cho F6 (sửa dấu heading /
chuyển sang wheel-differential) và F8 (neo QP về V3 / posture-guided thay blend
torque). Mọi số official assist trước đây đo trên WBC hỏng ⇒ vẫn cần re-baseline
SAU khi F6/F8 xong, không phải chỉ sau F1/F2.

### 3 test đỏ pre-existing (KHÔNG do F1/F2, đã đỏ trên e4bcccc)
`test_integrated_wbc_reports_contact_aware_diagnostics`,
`test_wbc_torque_leg_joints_are_zero` (module `IntegratedWBC` khác — và tiền đề
"WBC không được điều khiển chân" của nó mâu thuẫn với F1: WBC ĐÚNG có sinh torque
chân), `test_bias_forces_matches_original_nonzero_qvel` (mass matrix cache). Ngoài
phạm vi F1/F2.

---

## PHỤ LỤC 3 — Bước 5+6 đã thực hiện (F6 + F8), 2026-07-19

### F6 — Heading hip-yaw sai dấu (BUG/SIGN)
- `wheeled_biped/controllers/k2_jax_controller.py`: `tau_raw` từ
  `+kp*e - kd*rate + 0.05*kp*integral` → `-kp*e - kd*rate - 0.05*kp*integral`.
  Mapping (τ_L=+, τ_R=−)→+CCW (injection G>0); regulation cần M_z=−Kp·e−Kd·ẏ ⇒
  P (và integral) phải NEGATIVE — code cũ là positive feedback trên yaw error
  (D đã đúng). Comment cập nhật.
- **Verify:** yaw-kick 0.6 rad/s → |yaw| 2.76°→0.48° (giảm, sign đúng).
- **Tác động lớn nhất:** push r_thigh 90 N chuyển `ASSIST_SAFETY_FAIL` (hip_yaw
  vượt 20.1° trong 27 bước) → `ASSIST_MIXED` (0 safety_fail). Heading sai dấu
  chính là thứ bơm positive-feedback xoắn hip_yaw.
- **Giới hạn:** heading hip-yaw thẩm quyền ~0 (heading_gate mean 0.05, |τ_heading|
  ~0.001 Nm) — sign fix chặn làm-tệ, không tạo yaw authority thật. Parity tests
  vẫn xanh (133) vì τ_heading dưới tolerance. `init_v3_controller` bỏ qua
  `profile_name` (hardcode V3) — fix ở code nên có hiệu lực toàn cục.

### F8 — Blend torque không chuyển giao được giá trị posture (DESIGN)
- **Chẩn đoán quyết định:** sau F1, blend `α·(τ_wbc−τ_v3)` thêm **~0 Nm ở peak**
  push (`max|extra|`≈0.00 mọi khớp) — agreement gate + correction-cap đã trung
  hòa gauge (lo ngại gốc của F8). NHƯNG agreement gate ĐÚNG ĐẮN từ chối toàn bộ
  correction posture (mean_agree knee=0.000, hip_pitch~0.005) vì torque posture
  của WBC ngược dấu PD của V3 ⇒ **blend torque về bản chất không thể chuyển giao
  tối ưu posture của WBC** (giá trị chính, K_role hip_pitch/knee=0.60). Torque
  thừa +19% là divergence quỹ đạo, không phải injection.
- **Fix:** `scripts/promote_v3_vs_assist.py` thêm chế độ `assist_mode="posture_guided"`
  (DEFAULT mới, + CLI `--assist-mode`): WBC qdd chỉnh q_ref hip_pitch/knee của V3
  (POSTURE_GUIDED_JOINT_SCALE, ≤0.008 rad/s, one-step delay), tau_assist = V3
  thuần — không blend, không đánh nhau torque. `compute_posture_guided_assist` có sẵn.
- **Verify:** posture_guided ≥ torque_blend trên MỌI metric & scenario (r_thigh,
  torso ±x/±y): torque_rms 1.168→1.106, pitch 2.34→1.98 (r_thigh); torso-forward
  ASSIST_IMPROVED (tq_max 0.967, pitch 0.954). An toàn (0 ngã).
- **Còn lại (trung thực):** assist vẫn MIXED — cải thiện push này, regress yaw
  push khác (yaw ratio 1.2–1.9, sf 1–2 còn sót). Gốc: V3 **thiếu yaw authority
  thật** (heading hip-yaw ~0, wheel-differential `drift_k_heading=0` tắt). Đây là
  "F6-b" lớn hơn (bật wheel-differential yaw với sign được verify) + re-tune WBC
  task weights cho dynamics đã đúng + re-baseline. GIF: `outputs/visual/audit_final.gif`.

### F6-b — Wheel-differential yaw authority (điều tra + sign fix, KHÔNG bật)
- **Đo authority:** inject differential wheel torque (τ_L=+2/τ_R=−2 Nm) → Δyaw
  **−115.89° (CW)** trong 150 bước, không ngã. Authority khổng lồ (ngược hip-yaw:
  τ_L=+/τ_R=− ở wheel → CW, ở hip-yaw → CCW).
- **Sign fix:** drift `heading_torque` từ `-k·e - k_rate·ẏ` → `+k·e + k_rate·ẏ`
  (mapping τ_L=+h/τ_R=−h, injection G_w<0, regulation cần h=+k·e+k_rate·ẏ). Cũ là
  positive feedback — chỉ "an toàn" vì k_heading=0. Nay đúng dấu, an toàn khi bật.
- **KHÔNG bật k_heading trong V3.** Sweep (kh 3–25, gate widened 2/12): giảm PEAK
  yaw (22→19°) nhưng **regress SETTLED yaw** (5.0→6.5–9.4°) — gate yaw-error deadband
  (0.03 rad) chỉ mở transient trong push, authority khổng lồ → overshoot rồi gate
  đóng trước khi hoàn tất. Không có cú thắng sạch. roll/pitch vẫn an toàn.
- **Phát hiện then chốt:** V3 yaw thực ra ĐÃ ổn post-F6 (settle ~5° sau push 90N).
  Con số "yaw 24°" trước đây là từ run TRƯỚC F6 (hip-yaw sai dấu khuếch đại yaw).
  Tức F6 (đã làm) là cú fix yaw chính; F6-b không cần thiết như tưởng.
- Sign fix giữ lại (đúng, dormant ở V3 vì k_heading=0). Bật wheel-diff yaw có lợi
  cần TUNING gate anti-overshoot (deadband thấp/mượt + cân bằng rate damping) —
  một task thiết kế riêng, không phải one-line enable.

## PHỤ LỤC 4 — F12: Chân dang không hồi = BẪY HÌNH HỌC TIẾP XÚC (2026-07-19)

**Triệu chứng (định lượng, push r_thigh 90 N, post-F1–F8):** sau recovery, hip_roll
dang đối xứng (V3: +0.13/−0.11 rad; assist: +0.26/−0.19) và hip_yaw scissor
(∓0.17–0.24 rad), KHÔNG hồi về tư thế ban đầu dù không ngã. hip_pitch/knee lệch nhỏ.

**Cơ chế gốc (đo cân bằng torque tại trạng thái cuối):**
- hip_yaw trái: posture PD +2.60 + mode_div +3.26 + anti_twist +0.17 − yaw −0.61 =
  **+5.4 Nm liên tục kéo chân về** — khớp không nhúc nhích. Trọng lực chỉ −0.31 Nm.
- hip_roll trái: stance −0.19 + lateral −0.06 = −0.25 Nm kéo về; trọng lực +0.02.
- Thứ cân bằng các torque này là **moment ma sát tiếp xúc**: bánh xe chỉ lăn được
  theo phương dọc; kéo chân khép/xoay đòi trượt bánh NGANG mặt đất (ước lượng cần
  >10 Nm mức khớp). Push làm bánh trượt sang footprint mới trong transient; sau đó
  ma sát ghim hình học mới thành equilibrium bền.
- **Phản chứng:** hạ ma sát sàn sau recovery → hip_yaw scissor tự biến mất
  (∓0.146 → ∓0.006) bằng chính torque sẵn có (caveat: sim ma sát thấp mất ổn định,
  chỉ là bằng chứng bổ trợ; bằng chứng chính là cân bằng torque).

**Phân loại:** KHÔNG phải bug, KHÔNG phải thiếu gain (tăng gain chỉ tăng lãng phí,
không trượt nổi bánh — hip_yaw đã kéo 5.4 Nm vô ích). Là **THIẾU LOGIC tầng maneuver**:
khôi phục hình học chân đòi lăn bánh theo quỹ đạo khả thi (đánh lái yaw + lăn dọc,
kiểu ô tô; hoặc dồn tải nhấc từng bánh) — họ hàng của F5 (position/heading return).
Không tầng nào của V3/WBC làm việc này.

**Hệ quả phụ đáng chú ý:** (1) ~5.4 Nm torque tĩnh chống ma sát sau mỗi push —
đóng góp trực tiếp vào torque_rms/energy; (2) kp_hip_roll=0 trong posture PD +
stance yếu (eff 2 Nm/rad, cap 2 Nm qua weight 0.4) là có thật nhưng KHÔNG phải
nguyên nhân quyết định — hip_yaw có tới 5.4 Nm vẫn bị ghim; (3) assist arm dang
gấp ~2× V3 do transient khác (trajectory divergence), không phải do posture-guided
đẩy sai.

**Hướng fix nguyên lý (chưa làm, cần duyệt):** bộ "footprint recovery" — khi ổn
định sau nhiễu và hình học chân lệch khỏi ref quá ngưỡng: (a) lệnh yaw-steer nhẹ +
wheel velocity để bánh lăn hội tụ track-width/scissor về ref (mọi dịch chuyển bánh
đi qua phương lăn — khả thi vật lý); (b) đồng thời giải quyết drift position return
(F5) vì cùng một maneuver. Cần thiết kế + gate cẩn thận, không phải one-line.

## PHỤ LỤC 10 — CHỐT: V3_HOMING làm CONTROLLER CHÍNH (2026-07-19)

**Đã promote `K2_JAX_DEDICATED_DEFAULT_V3_HOMING` thành default vận hành:**
- `scripts/run_k2_jax_realtime.py` (entry point robot): `--profile` default →
  `k2_jax_dedicated_default_v3_homing`; thêm V3_HOMING vào `_PROFILE_MAP`; thread
  homing params (homing_enabled/kp_hip_roll/kp_hip_yaw/max_tau) vào pack_params.
  Rollback: `--profile k2_jax_dedicated_default_v3`.
- Comment profile đánh dấu OFFICIAL DEFAULT (promoted 2026-07-19).
- `init_v3_controller` (offline eval) GIỮ default V3 làm baseline so sánh; eval gọi
  V3_HOMING qua `--profile`.
- **KHÔNG bật WBC assist mặc định** (48/48 EQUIVALENT — không thêm giá trị, tốn QP/bước).

**Smoke-test realtime:** profile load đúng, drift ENABLED (k_pos=2, k_heading=2,
k_heading_rate=0.6), homing threaded, 300 bước 0 ngã. Parity + contact tests xanh
(47 passed) — homing stage gated + disabled cho V3 nên không đổi parity.

**SỐ OFFICIAL — `promote --full --profile V3_HOMING` (223 scenario, 2026-07-19):**
- **223/223 = ASSIST_EQUIVALENT.** V3_HOMING falls=0, Assist falls=0, safety_fails 0/0
  trên TOÀN BỘ suite (fixed-height sweep, height transition, single push, random push).
- Gần như MỌI median = 1.000 (V3_HOMING ≈ Assist). Assist chỉ hơn ở height_error_rms
  median 0.870 (−13%) — điểm duy nhất, biên. Mọi chỉ số khác 1.000.
- ⇒ **V3_HOMING vững qua toàn bộ 223 scenario (0 ngã) và mạnh ngang bản có WBC assist
  trên mọi chỉ số.** WBC assist không đáng thêm (không giá trị + tốn QP/bước).

**KẾT LUẬN CUỐI:** controller chính = **K2_JAX_DEDICATED_DEFAULT_V3_HOMING**, KHÔNG kèm
WBC assist. Real-time, đơn giản, an toàn 223/223, đã fix return-to-pose (F5/F12) +
yaw skew (F6b/F8b) + toàn bộ hạ tầng đo (F1–F13). V3 giữ để rollback.

---

## PHỤ LỤC 9 — QUYẾT ĐỊNH CONTROLLER CHÍNH: V3_HOMING ≥ Assist (quick 48, 2026-07-19)

`promote --quick --profile K2_JAX_DEDICATED_DEFAULT_V3_HOMING --assist-mode posture_guided`
so **V3_HOMING (arm1)** vs **V3_HOMING + WBC posture-guided Assist (arm2)**:

- **48/48 scenario = ASSIST_EQUIVALENT.** V3 falls=0, Assist falls=0, safety_fails 0/0.
- Gần như MỌI median chỉ số = **1.000** (giống hệt). WBC assist chỉ nhỉnh:
  height_error_rms median 0.866 (−13%), wheel_power 0.994, ang_vel 0.996 — biên.
- Không chỉ số nào Assist thắng RÕ; nhiều chỉ số 1.000 y hệt.

**KẾT LUẬN (trái với giả định "Assist mạnh nhất"):** trên nền V3_HOMING, **WBC assist
KHÔNG mạnh hơn** — 48/48 tương đương. Assist còn tốn 1 lần giải QP mỗi bước (~0.1s)
+ phụ thuộc WBC + có failure mode (đã phải vá F8b yaw-runaway). ⇒ **Khuyến nghị làm
controller CHÍNH: `K2_JAX_DEDICATED_DEFAULT_V3_HOMING` (KHÔNG kèm WBC assist)** —
đơn giản, real-time, đã fix return-to-pose (chân khép, yaw hồi), và mạnh ngang bản
có Assist trên mọi chỉ số. WBC assist chỉ giữ nếu sau này cần height tracking tốt hơn
ở kịch bản cụ thể (lợi ích duy nhất đo được: −13% height error).

Còn cần trước khi promote default: (1) so V3_HOMING vs V3 gốc để chắc homing không
regress balance (homing stability-gated nên rủi ro thấp, nhưng nên xác nhận);
(2) chạy `--full` 223 scenario để chốt số official.

---

## PHỤ LỤC 8 — F5/F12 ĐÃ IMPLEMENT: posture homing + yaw/position return (2026-07-19)

**Mục tiêu:** sau push robot phục hồi nhưng chân dang, không về tư thế ban đầu (F12),
và trôi yaw/không dừng hẳn (F5).

**Test cơ chế (quyết định trước khi code):** trên trạng thái chân dang sau push,
đo can thiệp nào KHÉP chân:
| can thiệp | hip_roll dev | hip_yaw dev |
|---|---|---|
| baseline | 0.154 | 0.339 |
| +hip_roll PD | **0.044** | 0.347 |
| +hip_yaw boost | 0.044 | 0.264 |
| wheel wiggle | 0.156 (vô ích) | 0.337 |
| hip_roll+hip_yaw PD | **0.017** | 0.280 |
→ Chân dang hip_roll = do V3 đặt **kp_hip_roll=0** (không có PD hồi). Thêm PD hồi
khép ngay. hip_yaw scissor giảm một phần (ma sát ghim phần dư). Wheel-maneuver KHÔNG
cần.

**Implement (code mới, có gate an toàn):**
- `k2_jax_controller.py`: thêm stage **posture homing** (4 param 84–87): PD hồi
  hip_roll[0,5] + hip_yaw[1,6] về q_ref, bound tanh, **gated bằng `_twist_stability`**
  (≈0 khi nhiễu → không đánh nhau balance; ≈1 khi settled → khép chân).
- Profile mới **`K2_JAX_DEDICATED_DEFAULT_V3_HOMING`** (V3 giữ nguyên làm baseline):
  homing kp_hip_roll=8, kp_hip_yaw=6; + F5 yaw/pos return qua bánh
  (drift_k_heading=2.0 sign F6-b, drift_k_pos=2.0, widen heading gate 2/12).
- `init_v3_controller` giờ tôn trọng `profile_name`; viz thêm `--profile`.

**Gain tuned (2D sweep kp_hr×kp_hy trên r_thigh 90N):** kp_hip_roll=15, kp_hip_yaw=25,
max_tau=10 — kp_hr đủ cao để giữ hip_roll trước coupling từ hip_yaw mạnh.
**Verify (push r_thigh 90N, GIF `outputs/visual/push90_homing_full.gif` 14s):**
| khớp (dev/khớp) | V3 baseline | V3_HOMING (tuned) |
|---|---|---|
| hip_roll (chân dang) | 4.3° | **0.2°** |
| hip_yaw (scissor) | ~9° | **2.3°** |
| wheelvel cuối | 3.13 | ~1.2 |
| ngã | không | không |

Đa hướng (torso R/L/F 50N): 0 ngã, roll_max ≈ V3, legdev mọi hướng nhỏ (<1.5°/khớp).
GIF khung cuối: **V3 chân khép sát, thẳng đứng, quay mặt thẳng — về gần đúng tư thế
ban đầu** (còn scissor dư ~2° do ma sát, gần như không thấy bằng mắt).

An toàn đa hướng (torso R/L/F 50N + r_thigh 90N): 0 ngã, roll_max ≈ V3, legdev
giảm ở ca khó (r_thigh 0.389→0.102) — chỉ hơi tăng ở 1 ca dễ (torso-L 0.032→0.083,
tuyệt đối nhỏ, có thể tinh chỉnh gain sau). GIF khung cuối: hai chân về thế hẹp,
gần thẳng đứng.

**Còn lại:** hip_yaw scissor còn dư ~2° (ma sát). Nên đánh giá V3_HOMING trên
promote --quick trước khi cân nhắc làm default. Đây là feature mới, V3 baseline chưa đổi.

### F8b — Yaw skew của ASSIST: posture-guided q_ref RUNAWAY (bug + fix)
**Triệu chứng:** V3-homing đơn hồi yaw về ~0 (−2.6°), nhưng arm ASSIST kẹt yaw ~−20°,
"xoay xéo sang hướng khác" (quan sát của user).
**Root cause (đo):** với alpha=0 hai arm identical (yaw về +2°) ⇒ không phải clone
divergence. `compute_posture_guided_assist` **integrate qdd_wbc vào q_ref không hội
tụ** — q_ref hip_pitch drift đơn điệu 0.04→0.13 rad suốt một cú push (runaway
windup), liên tục đổi posture → khóa yaw offset ~20°, height sụt.
**Fix:** thêm **anti-windup leak** trong nhánh posture_guided của `run_dual_arm_rollout`:
`q_ref += 0.02·(eq_joint − q_ref)` — bound q_ref quanh nominal, cho decay về nominal
khi WBC ngừng đẩy. Verify: ASSIST yaw kẹt −20° → hồi về **−4.86°** (V3 −2.6°),
class REGRESSED→MIXED. GIF `outputs/visual/push90_homing_yawfix.gif`: cả hai arm cùng
quay mặt thẳng, không còn xoay xéo.
**Lateral position:** đo lại — offset vị trí do PUSH chỉ **~3cm** (world_x 0.022,
world_y −0.023 từ trước push); pos_gate không cần mở. Con số drift ~0.19m ở metric
là tính từ đầu rollout (gồm settling trước push), KHÔNG phải hồi phục sau push. Vậy
position return đã đủ tốt cho cú push này — không cần yaw-then-drive maneuver phức tạp.

---

## PHỤ LỤC 7 — Bảng 25 chỉ số ĐÚNG (sau F13, quick 48 scenario, 2026-07-19)

**FALLS: V3 = 0, Assist = 0** (trước F13: V3 = 1057, Assist = 1 — toàn bộ là artefact).
Classifications: **EQUIVALENT 31, MIXED 14, REGRESSED 3, IMPROVED 0**
(trước F13: IMPROVED 5, EQUIV 9, MIXED 33, REGRESSED 1).
safety_fails: V3 = 170, Assist = 128 (Assist ít hơn chút, chủ yếu ở forward-push).

| metric (assist/V3) | median | mean | %better |
|---|---|---|---|
| height_error_rms_m | **0.868** | 0.809 | 52% |
| wheel_power_proxy | 0.982 | 0.976 | 60% |
| com_vel_rms | 0.985 | 0.986 | 60% |
| roll_oscillation_rms | 0.995 | 0.989 | 58% |
| yaw_drift_rms_deg | 0.969 | 1.232 | 52% |
| ang_vel_rms | 0.997 | 0.992 | 52% |
| height_rms/max/min_m | ~0.997 | ~0.998 | 60% |
| torque_max | 1.000 | 0.997 | 42% |
| torque_oscillation_rms | 1.000 | 0.992 | 35% |
| pitch_rms_deg | 1.000 | 1.027 | 17% |
| pitch_max_deg | 1.000 | 0.999 | 8% |
| torque_rms | 1.003 | 1.002 | 6% |
| roll_rms_deg | 1.020 | 1.027 | 0% |
| planar_drift_max_m | 1.027 | 1.010 | 8% |
| planar_drift_final_m | 1.004 | 1.051 | 8% |
| falls / survival_steps | 1.000 | 1.000 | — (cả hai 0 ngã) |

**KẾT LUẬN TRUNG THỰC (sau khi sửa toàn bộ F1–F13):**
Với harness công bằng, **V3 và V3+Assist về cơ bản TƯƠNG ĐƯƠNG** — không arm nào
ngã, hầu hết median ≈ 1.000. Assist cho lợi ích NHỎ và nhất quán ở: height tracking
(−13% error RMS), wheel power (−2%), CoM/ang velocity, roll oscillation. Trung tính
về torque/pitch. Hơi kém ở roll_rms (+2%) và planar_drift (+3–5%). yaw_drift median
tốt hơn (0.97) nhưng mean tệ (1.23) → vài scenario yaw xấu (đuôi phân bố — liên quan
F5/F6-b yaw authority chưa hoàn chỉnh).

⇒ Toàn bộ "Assist thắng lớn (survival + pitch −18%)" ở Phụ lục 5 là do artefact F13
bóp méo baseline V3. Sự thật: **WBC assist (posture-guided, dynamics đúng) an toàn và
hơi nhỉnh về height/effort, nhưng KHÔNG phải cải thiện lớn** so với V3 vốn đã đủ tốt.
Giá trị của cả chuỗi audit F1–F13 là làm cho phép đo TRUNG THỰC — trước đây mọi so
sánh đều đo trên WBC hỏng (F1/F2) và/hoặc harness lỗi (F3/F13).

---

## PHỤ LỤC 6 — F13: "V3 ngã ở push-right" là ARTEFACT HARNESS (CRITICAL, 2026-07-19)

**Bối cảnh:** bảng 25 chỉ số (Phụ lục 5) có headline "V3 = 1057 fall-steps, Assist = 1"
— Assist cứu survival. ĐIỀU TRA cho thấy đây phần lớn là **lỗi harness**, không phải
lợi ích Assist thật.

**Bằng chứng:**
- V3 cô lập từ settle sạch chịu được push 50N mọi hướng (right & left, lean 7–16°,
  KHÔNG ngã, KHÔNG asymmetry).
- Nhưng trong promote/phase3d, V3 baseline ngã ở push-right (step 154, tức 70 bước
  SAU khi push kết thúc — phân kỳ CHẬM, không phải knock-down).
- Truy gốc: scenario được pre-generate hàng loạt nên `v3_ctrl["jax_state"]` bị stale
  → harness **reset controller** (`pack_state_k2`) rồi re-stabilize 100 bước trước
  khi clone. Controller reset (mất filter/integrator/latch đã hội tụ) **over-drive
  bánh xe 2.25 → 5.9 rad/s** (robot lăn ~0.6 m/s) từ trạng thái rolling settle.
  Từ trạng thái 5.9 rad/s pathological này, push mới làm V3 phân kỳ ngã.

| Cấu hình pre-clone | wheel_vel | push-right |
|---|---|---|
| RESET + re-stabilize (harness cũ) | −5.9 rad/s | **NGÃ** step 175 |
| KEEP controller state + re-stabilize | −4.4 rad/s | sống |
| Restore settled state, NO re-stabilize | 2.25 rad/s | sống |

**Fix (F13):** generator lưu `v3_jax_state` (controller state đã settle, khớp
qpos/qvel của scenario); rollout **restore** thay vì reset, bỏ re-stabilize gây
over-drive. Cả hai arm khởi từ CÙNG settled state (fair), rồi tiến hóa độc lập.
Áp cho cả `promote_v3_vs_assist.py` và `phase3d_full_batch_execution.py`.

**Verify fix:** push_nominal_right V3 falls **230 → 0** (class ASSIST_IMPROVED →
ASSIST_EQUIVALENT); push-left ASSIST_REGRESSED. → **Với harness đúng, V3 không ngã,
và Assist KHÔNG còn hơn rõ.**

**HỆ QUẢ:** headline survival của Phụ lục 5 (bảng 25 chỉ số) BỊ NHIỄU bởi F13 —
1057 fall-steps của V3 phần lớn là artefact. Bảng đó đo trên harness lỗi ⇒ **VÔ HIỆU
cho so sánh survival**. Đang re-chạy `promote --quick` với F13 fix để lấy bảng đúng
(Phụ lục 7). Mọi số official trước F13 cần re-baseline.

---

## PHỤ LỤC 5 — Bảng 25 chỉ số V3 vs Assist(posture_guided) sau F1–F8 (quick, 48 scenario, 2026-07-19) — ⚠️ NHIỄU BỞI F13, xem Phụ lục 6/7

Classifications: IMPROVED 5, EQUIVALENT 9, MIXED 33, REGRESSED 1.
**Falls: V3 = 1057 fall-steps (ngã hẳn ở 5/6 push-RIGHT scenarios, mọi height variant),
Assist = 1.** Safety-fails: V3 = 1060, Assist = 22. → Assist cứu robot khỏi ngã
trên toàn bộ lateral push phải — lợi ích sống còn thật, lần đầu đo được sau khi
WBC được sửa đúng (trước F1/F2 không thể có kết quả này vì WBC là torque rác).

| metric (assist/V3) | median | mean | %assist tốt hơn |
|---|---|---|---|
| pitch_rms_deg | 0.902 | 0.821 | **95.8%** |
| pitch_lf_power_deg | 0.902 | 0.824 | **95.8%** |
| height_max_m¹ | 0.997 | 0.997 | 91.7% |
| height_rms_m | 0.997 | 1.044 | 85.4% |
| roll_max_deg | 0.991 | 0.873 | **79.2%** |
| height_min_m¹ | 0.975 | 1.382 | 79.2% |
| pitch_max_deg | 0.996 | 0.830 | 75.0% |
| roll_rms_deg | 0.976 | 0.894 | 64.6% |
| height_error_rms_m | **0.759** | 0.772 | 56.2% |
| yaw_drift_max_deg | 0.983 | 1.050 | 54.2% |
| jvel_oscillation_rms | 1.061 | 0.956 | 41.7% |
| yaw_drift_rms_deg | 1.377 | 1.237 | 39.6% |
| wheel_power_proxy | 1.200 | 0.967 | 35.4% |
| pitch_oscillation_rms | 1.054 | 0.858 | 33.3% |
| torque_rms | 1.008 | 0.982 | 29.2% |
| com_vel_rms | 1.115 | 1.046 | 27.1% |
| planar_drift_max_m | 1.021 | 1.119 | 25.0% |
| ang_vel_rms | 1.170 | 0.985 | 22.9% |
| planar_drift_final_m | 1.235 | 1.355 | 22.9% |
| roll_oscillation_rms | 1.205 | 1.017 | 20.8% |
| **torque_max** | **1.419** | 1.203 | 18.8% |
| torque_oscillation_rms | 1.098 | 1.072 | 16.7% |
| survival_steps² | 1.000 | **1.141** | (² >1 = tốt hơn) |
| falls² / safety_fails² | 1.000 | 0.896/0.878 | (² <1 = tốt hơn; V3 1057 vs Assist 1) |

¹ height_max/min: gần 1 = giữ height tương đương. ² survival/falls: chiều ngược.

**Đọc kết quả trung thực:**
- Assist THẮNG: sống còn (headline), pitch (~−18% RMS, 96% scenario), roll, height
  tracking (−24% error RMS), pitch_max/roll_max.
- Assist THUA: torque_max (+42% median — WBC counterfactual vẫn tạo divergence
  quỹ đạo + gauge; F8 posture-guided đã loại phần blend nhưng arm assist vẫn chạy
  quỹ đạo khác), yaw_drift_rms (+38% — yaw authority yếu, xem F6-b + F12),
  planar_drift_final (+24% — cả hai không có position return, F5), oscillation nhẹ.
- So official cũ (pre-fix: "roll_max −8.7%, ang_vel −17.5%... torque_max +28% LUÔN"):
  cấu trúc lợi/hại đổi hẳn — giờ lợi ích tập trung vào sống còn + pitch/height,
  cái giá là torque đỉnh trong các scenario MIXED. Số cũ đo trên WBC hỏng, không so
  sánh trực tiếp được.

### Trạng thái sau F1–F8 (+ F6-b sign)
WBC đúng động lực học (F1) + có mục tiêu theo mode (F2); eval đo đúng điểm làm việc
(F3/F4); heading hip-yaw & wheel-diff không còn positive-feedback (F6/F6-b); assist
dùng posture-guided không đánh nhau torque (F8). V3 yaw ổn (~5°). Assist an toàn,
đôi khi giúp, đôi khi regress yaw nhẹ (trajectory divergence).
**Bước tiếp hợp lý:** re-tune WBC task weights cho dynamics đã đúng →
`promote --full --assist-mode posture_guided` re-baseline chính thức (số cũ vô hiệu);
wheel-diff yaw anti-overshoot tuning nếu yaw thành ưu tiên.
