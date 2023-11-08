Trong file test có thông tin như sau:


    - dòng 1: góc bao phủ của 1 sensor

    - dòng 2: bán kính

    - dòng 3: kích thước miền quan sát

    - dòng 4: số target: n

    - dòng 5: số sensor: m

    - n dòng tiếp theo: (x,y, a) trong đó x,y là tọa độ, a là số sensor cần bao phủ

    - m dòng tiếp theo: tọa độ sensor



File kết quả:

    - các góc alpha của vector f(i,j)

    - metric đạt được:

        -   chỉ số CQ: Coverage quantities

        -   chỉ số QBI: Q- balance index

        -   số sensor tối thiểu sử dụng thỏa mã Q- cover

Chạy code:

- chạy code ra kq:
python3 nonplaceQcover/main.py

  - thay đổi nội dung file:
    if __name__ == '__main__':
        solve("test/test.inp")
    để thay đổi test case; hoặc uncomment đoạn code trong main để chạy hết test case.
    
- Để visualize kết quả có thể vào file draw.ipynb thay đổi đường dẫn input và kq tương đương:
result_path = "results/results.json"
target_path = "test/test.inp"
sau đó chạy các ô bên trong
