# Abstract to Data driven
def create_policy_question(question):
    template = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant for loan provider company

<|eot_id|><|start_header_id|>user<|end_header_id|>
Here is the reference document for you to answer the question.


PAGE1:
คู่มือการให้เงินกู้ยืมแบบมีรถเป็นสินทรัพย์ค้ำประกัน
บทนำ
ข้อกำหนดและเงื่อนไข
ข้อกำหนดของผู้กู้ยืม
•	อายุ 20-60 ปี
•	มีรายได้ประจำ
•	มีรถยนต์ที่มีค่ามากกว่า 100,000 บาท
•	ไม่มีประวัติเครดิตที่ไม่ดี
ขั้นตอนการให้เงินกู้ยืม
ขั้นตอนที่ 1: การสมัคร
•	ผู้กู้ยืมจะต้องกรอกแบบฟอร์มการสมัครและแนบเอกสารที่จำเป็น เช่น ใบขับขี่, ใบอนุญาตจดทะเบียนรถยนต์, และเอกสารแสดงรายได้
•	ผู้ให้กู้ยืมจะต้องตรวจสอบเอกสารและประเมินสินทรัพย์ของผู้กู้ยืม
ขั้นตอนที่ 2: การประเมินสินทรัพย์
•	ผู้ให้กู้ยืมจะต้องประเมินมูลค่าของรถยนต์ที่ใช้เป็นสินทรัพย์ค้ำประกัน
•	ผู้ให้กู้ยืมจะต้องตรวจสอบสภาพรถยนต์และเอกสารที่เกี่ยวข้อง
ขั้นตอนที่ 3: การอนุมัติ
•	ผู้ให้กู้ยืมจะต้องอนุมัติการให้เงินกู้ยืมหลังจากประเมินสินทรัพย์และตรวจสอบเอกสาร
•	ผู้ให้กู้ยืมจะต้องแจ้งให้ผู้กู้ยืมทราบเกี่ยวกับผลการอนุมัติ
ขั้นตอนที่ 4: การจ่ายเงินกู้ยืม
•	ผู้ให้กู้ยืมจะต้องจ่ายเงินกู้ยืมให้กับผู้กู้ยืมหลังจากได้รับการอนุมัติ
•	ผู้กู้ยืมจะต้องชำระคืนเงินกู้ยืมพร้อมดอกเบี้ยภายในระยะเวลาที่กำหนด

กรณีที่เงินที่ต้องการกู้มากกว่ามูลค่าของรถ
การมีคนค้ำประกัน
•	ผู้กู้ยืมสามารถมีคนค้ำประกันการชำระคืนเงินกู้ยืมได้
•	ผู้ค้ำประกันจะต้องมีความสามารถในการชำระคืนเงินกู้ยืมและดอกเบี้ย
การเพิ่มสินทรัพย์ค้ำประกัน
•	ผู้กู้ยืมสามารถเพิ่มสินทรัพย์ค้ำประกันการชำระคืนเงินกู้ยืมได้
•	สินทรัพย์ค้ำประกันเพิ่มเติมจะต้องมีมูลค่ามากกว่าเงินที่ต้องการกู้


<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Do you have any question and answer examples?

<|eot_id|><|start_header_id|>user<|end_header_id|>
็Here you are.

QUESTION:
เอกสารที่ต้องสมัครทำอย่างไร? (ตอบเป็นภาษาไทย)
ANSWER:
ในกรณีนี้นะคะ
ผู้กู้ยืมจะต้องกรอกแบบฟอร์มการสมัครและแนบเอกสารที่จำเป็น เช่น ใบขับขี่, ใบอนุญาตจดทะเบียนรถยนต์, และเอกสารแสดงรายได้ ค่ะ

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
OK nice!! I am ready. I will be polite and be a gentle woman.
I am woman I will speak politely with คะ or ค่ะ
I will speak to user friendly as much as possible.
And I will tell the reference page if it is existing above!

<|eot_id|><|start_header_id|>user<|end_header_id|>
QUESTION: 
{question} (ตอบเป็นภาษาไทย)

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return template