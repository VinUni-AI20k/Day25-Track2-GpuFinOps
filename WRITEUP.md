# Lab 25 — GPU FinOps Optimization: Bài viết tổng hợp

**Sinh viên:** Cao Minh Quang (2A202601884)
**Ngày:** 27/08/2026
**Vai trò giả định:** FinOps Engineer @ NimbusAI

---

## 1. Baseline vs. Optimized

| | Baseline | Optimized | Tiết kiệm |
|---|---:|---:|---:|
| **Tổng chi phí (tháng)** | $27,133 | $14,626 | **$12,507 (46%)** |
| **Inference — $/1M-token** | $6.488 | $1.126 | **82.6%** |
| **Purchasing (spot/reserved)** | $25,667 (on-demand) | $15,627 | 39.1% |

Con số quan trọng nhất không phải "$/GPU-giờ" mà là **$/1M-token**: hai đội trả cùng giá
GPU nhưng đội tối ưu tốt hơn phục vụ nhiều token hơn trên cùng một đồng chi. Ở đây,
`$/1M-token` giảm từ $6.488 xuống $1.126 — tức phục vụ cùng lượng token với **chưa tới
1/5 chi phí ban đầu**.

## 2. Phân tích từng đòn bẩy

**M5 — 4 đòn bẩy đóng góp vào $12,507 tiết kiệm:**

| Lever | Savings ($/tháng) | % tổng savings |
|---|---:|---:|
| **Purchasing (spot/reserved)** | **$10,040** | **80.3%** |
| Inference (cascade/cache/batch) | $1,212 | 9.7% |
| Right-size util-lies | $655 | 5.2% |
| Kill idle GPUs | $600 | 4.8% |

Purchasing (chuyển đúng workload sang spot hoặc reserved) là đòn bẩy **lớn nhất về số tiền
tuyệt đối** — vì đây tác động lên toàn bộ hóa đơn GPU-giờ, trong khi 3 lever còn lại chỉ tác
động lên phần inference hoặc phần idle nhỏ hơn.

Tuy nhiên, khi tách riêng nội bộ đòn bẩy inference (M2) thành 3 phần bật/tắt độc lập:

| Lever inference (bật riêng lẻ) | Savings so baseline |
|---|---:|
| **Cascade** (route model nhỏ) | **76.5%** |
| Batch API | 16.9% |
| Prompt caching | 9.4% |
| Cả 3 kết hợp | 82.6% |

→ Trong nội bộ chi phí inference, **cascade là lever mạnh nhất, áp đảo hoàn toàn** —
đơn giản vì phần lớn traffic đủ điều kiện dùng model nhỏ rẻ hơn ~15×, còn cache/batch chỉ
"vét" thêm phần chi phí còn sót lại sau khi đã cascade.

## 3. GPU-Util Lie

Hai GPU bị gắn cờ "nói dối" (`util ≥ 90%` nhưng `MFU < 30%`):

| GPU | Util% | MFU |
|---|---:|---:|
| `gpu-h100-4` | 98.2% | **0.194** |
| `gpu-a10g-1` | 96.9% | 0.268 |

`nvidia-smi` GPU-Util chỉ đo "SM có đang bận tại thời điểm lấy mẫu", kể cả khi phần lớn
thời gian đó là memory-stall hoặc I/O wait chứ không sinh FLOPs hữu ích. MFU (FLOPs thực
đạt / FLOPs đỉnh) mới phản ánh đúng "tiền bỏ ra mua được gì".

**Tác động tài chính:** M5 quy đổi việc hạ tier các GPU bị "lie" này xuống một bậc
(H100→A100) tiết kiệm **$655/tháng**. Ngoài ra `gpu-h100-5` (không bị lie, nhưng có 8h/ngày
idle) gây lãng phí **$600/tháng** (≈2.2% tổng chi phí baseline) — GPU vẫn được tính tiền dù
không chạy job nào.

## 4. Phần mở rộng đã thực hiện

### Extension 3 — `cache_is_worth_it()`
Thêm hàm vào `finops/pricing.py` xác định caching có đáng làm không dựa trên số lần đọc
lại trung bình của một prefix. Mô hình: ghi cache tốn bằng giá đọc thường (không lời gì),
mỗi lần đọc lại sau đó chỉ tốn `read_discount × giá thường`. Điểm hòa vốn suy ra:
`avg_cache_reads > 1/(1 - read_discount)`.

**Đo lường:** với discount mặc định 10% (giảm 90%), chỉ cần > 1.11 lần đọc lại là cache đã
có lời — tức hầu hết prefix được tái sử dụng (system prompt, RAG context phổ biến) đều nên
cache. Ngược lại, prefix chỉ dùng đúng 1 lần (`avg_cache_reads=1`) thì **không bao giờ hòa
vốn**, bất kể discount sâu đến đâu. Insight: discount 90% khiến ngưỡng hòa vốn cực thấp —
lỗi thường gặp không phải "cache có lời không" mà là "có thực sự tái sử dụng prefix không".

### Extension 4 — Ngân sách Reasoning
Sửa `missions/m2_inference_levers.py` (hàm `reasoning_budget()`) và
`missions/m5_report.py` để tách riêng $ và Wh cho traffic `is_reasoning=1` vs `=0`.

**Đo lường (số liệu thật từ 2,400 request):**

| | Reasoning | Non-reasoning |
|---|---:|---:|
| Requests | 201 (8.4%) | 2,199 (91.6%) |
| Tokens | 1.24M (16.5%) | 6.29M |
| Cost/ngày | $1.40 (16.5%) | $7.09 |
| **Energy/ngày** | **29,788 Wh (94.0%!)** | 1,888 Wh |
| Wh/query | 148.2 | 0.86 |

**Insight quan trọng nhất:** reasoning chỉ chiếm 8.4% request và 16.5% chi phí, nhưng ngốn
**94% tổng năng lượng inference** (hệ số ~80× mỗi query). Đây là đòn bẩy sustainability lớn
nhất chưa được các mission M1–M3 chạm tới — kiểm soát cost `$` là chưa đủ, cần kiểm soát cả
`Wh`/carbon. Đề xuất: gate reasoning tier sau ngưỡng confidence thấp (< 0.6) từ model nhanh
hoặc cờ high-stakes rõ ràng; dataset hiện chưa có cột confidence nên bước đầu tiên là
instrument nó trước khi áp quy tắc routing tự động.

## 5. Khuyến nghị cho NimbusAI (3 hành động đầu tiên nếu là FinOps lead)

1. **Xử lý ngay 2 "quick win" rẻ tiền, rủi ro thấp:** tắt/hạ tier `gpu-h100-4`,
   `gpu-a10g-1` (util-lie) và `gpu-h100-5` (idle 8h/ngày) — thu về **~$1,255/tháng**
   không cần đổi hành vi của bất kỳ team nào, làm được trong tuần đầu tiên.
2. **Bật cascade routing toàn công ty cho traffic inference** — đây là lever mạnh nhất
   trong nội bộ chi phí inference (76.5% savings riêng lẻ). Trước khi bật đại trà, audit
   traffic hiện đang gọi thẳng model lớn để xác nhận đủ điều kiện route xuống model nhỏ
   mà không ảnh hưởng chất lượng.
3. **Đàm phán reserved cho 3 job inference ổn định** (`job-infer-chat`, `job-infer-rag`,
   `job-infer-search`) và chuyển toàn bộ job training/dev có thể gián đoạn sang
   spot+checkpoint — đây là lever lớn nhất về số tiền tuyệt đối ($10,040/tháng, 80.3% tổng
   savings). Cần xác minh utilization thực tế ≥ 55% (điểm hòa vốn) trước khi ký cam kết
   3 năm, tránh trả tiền cho capacity dư thừa nếu traffic sau này giảm.

*Theo dõi dài hạn: instrument confidence score cho reasoning traffic (Ext 4) để chuyển đề
xuất routing từ định tính sang dựa trên dữ liệu thật.*
