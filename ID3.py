import numpy as np
import pandas as pd
from itertools import combinations


# ============================================================
# PHẦN 1: THUẬT TOÁN ID3 - DỰ ĐOÁN CHƠI BÓNG
# ============================================================

weather_data = {
    'Ma': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],

    'ThoiTiet': [
        'sunny', 'sunny', 'overcast', 'rainy',
        'rainy', 'rainy', 'overcast', 'sunny',
        'sunny', 'rainy', 'sunny', 'overcast',
        'overcast', 'rainy'
    ],

    'NhietDo': [
        'hot', 'hot', 'hot', 'mild',
        'cool', 'cool', 'cool', 'mild',
        'cool', 'mild', 'mild', 'mild',
        'hot', 'mild'
    ],

    'DoAm': [
        'high', 'high', 'high', 'high',
        'normal', 'normal', 'normal', 'high',
        'normal', 'normal', 'normal', 'high',
        'normal', 'high'
    ],

    'Gio': [
        'weak', 'strong', 'weak', 'weak',
        'weak', 'strong', 'strong', 'weak',
        'weak', 'weak', 'strong', 'strong',
        'weak', 'strong'
    ],

    'ChoiBong': [
        'no', 'no', 'yes', 'yes',
        'yes', 'no', 'yes', 'no',
        'yes', 'yes', 'yes', 'yes',
        'yes', 'no'
    ]
}

bang_thoi_tiet = pd.DataFrame(weather_data).set_index('Ma')

print("========== DỮ LIỆU THỜI TIẾT ==========")
print(bang_thoi_tiet)


# Hàm tính Entropy
def tinh_entropy(cot_muc_tieu):
    gia_tri, so_luong = np.unique(
        cot_muc_tieu,
        return_counts=True
    )

    xac_suat = so_luong / len(cot_muc_tieu)

    return -np.sum(
        xac_suat * np.log(xac_suat)
    )


# Hàm tính Information Gain
def tinh_information_gain(data_frame, thuoc_tinh, muc_tieu='ChoiBong'):

    entropy_goc = tinh_entropy(
        data_frame[muc_tieu]
    )

    gia_tri, so_luong = np.unique(
        data_frame[thuoc_tinh],
        return_counts=True
    )

    entropy_sau_khi_chia = 0

    for i in range(len(gia_tri)):

        tap_con = data_frame[
            data_frame[thuoc_tinh] == gia_tri[i]
        ]

        entropy_sau_khi_chia += (
            so_luong[i] / np.sum(so_luong)
        ) * tinh_entropy(
            tap_con[muc_tieu]
        )

    gain = entropy_goc - entropy_sau_khi_chia

    return gain, entropy_sau_khi_chia


cac_thuoc_tinh = [
    'ThoiTiet',
    'NhietDo',
    'DoAm',
    'Gio'
]

muc_tieu = 'ChoiBong'

print("\n========== ID3 - ENTROPY ==========")

entropy_ban_dau = tinh_entropy(
    bang_thoi_tiet[muc_tieu]
)

print(
    f"Entropy ban đầu H(S) = {entropy_ban_dau:.4f}"
)

bang_gain = []

for cot in cac_thuoc_tinh:

    gain, entropy_chia = tinh_information_gain(
        bang_thoi_tiet,
        cot,
        muc_tieu
    )

    bang_gain.append({
        'Thuoc tinh': cot,
        'H(X,S)': round(entropy_chia, 4),
        'Information Gain': round(gain, 4)
    })

bang_ket_qua = pd.DataFrame(bang_gain)

print(
    bang_ket_qua.sort_values(
        by='Information Gain',
        ascending=False
    )
)


# Hàm xây dựng cây ID3
def xay_dung_id3(
    du_lieu,
    du_lieu_goc,
    danh_sach_thuoc_tinh,
    muc_tieu='ChoiBong',
    lop_cha=None
):

    if len(du_lieu) == 0:

        gia_tri, so_luong = np.unique(
            du_lieu_goc[muc_tieu],
            return_counts=True
        )

        return gia_tri[np.argmax(so_luong)]

    elif len(np.unique(
        du_lieu[muc_tieu]
    )) == 1:

        return np.unique(
            du_lieu[muc_tieu]
        )[0]

    elif len(danh_sach_thuoc_tinh) == 0:

        return lop_cha

    else:

        gia_tri, so_luong = np.unique(
            du_lieu[muc_tieu],
            return_counts=True
        )

        lop_cha = gia_tri[
            np.argmax(so_luong)
        ]

        danh_sach_gain = []

        for thuoc_tinh in danh_sach_thuoc_tinh:

            gain, _ = tinh_information_gain(
                du_lieu,
                thuoc_tinh,
                muc_tieu
            )

            danh_sach_gain.append(gain)

        vi_tri_max = np.argmax(
            danh_sach_gain
        )

        thuoc_tinh_tot_nhat = (
            danh_sach_thuoc_tinh[vi_tri_max]
        )

        cay = {
            thuoc_tinh_tot_nhat: {}
        }

        thuoc_tinh_con_lai = [
            x for x in danh_sach_thuoc_tinh
            if x != thuoc_tinh_tot_nhat
        ]

        for gia_tri_thuoc_tinh in np.unique(
            du_lieu[thuoc_tinh_tot_nhat]
        ):

            tap_con = du_lieu[
                du_lieu[thuoc_tinh_tot_nhat]
                == gia_tri_thuoc_tinh
            ]

            cay[
                thuoc_tinh_tot_nhat
            ][gia_tri_thuoc_tinh] = xay_dung_id3(
                tap_con,
                du_lieu_goc,
                thuoc_tinh_con_lai,
                muc_tieu,
                lop_cha
            )

        return cay


def in_cay_id3(cay, khoang_cach=""):

    if not isinstance(cay, dict):

        print(
            f" ──> [Choi bong = {cay.upper()}]"
        )

        return

    for thuoc_tinh, nhanh in cay.items():

        for gia_tri, cay_con in nhanh.items():

            print(
                f"{khoang_cach}├── "
                f"({thuoc_tinh} == '{gia_tri}')",
                end=""
            )

            if isinstance(cay_con, dict):

                print()

                in_cay_id3(
                    cay_con,
                    khoang_cach + "│   "
                )

            else:

                print(
                    f" ──> "
                    f"[Choi bong = {cay_con.upper()}]"
                )


cay_thoi_tiet = xay_dung_id3(
    bang_thoi_tiet,
    bang_thoi_tiet,
    cac_thuoc_tinh
)

print("\n========== CÂY QUYẾT ĐỊNH ID3 ==========")

in_cay_id3(cay_thoi_tiet)


# Hàm dự đoán
def du_doan_id3(cay, mau):

    if not isinstance(cay, dict):

        return cay

    thuoc_tinh = list(cay.keys())[0]

    gia_tri = mau.get(
        thuoc_tinh
    )

    cay_con = cay[
        thuoc_tinh
    ].get(gia_tri)

    if isinstance(cay_con, dict):

        return du_doan_id3(
            cay_con,
            mau
        )

    return cay_con


mau_thu = {
    'ThoiTiet': 'sunny',
    'NhietDo': 'cool',
    'DoAm': 'high',
    'Gio': 'weak'
}

ket_qua_du_doan = du_doan_id3(
    cay_thoi_tiet,
    mau_thu
)

print("\n========== DỰ ĐOÁN ==========")

print("Dữ liệu:", mau_thu)

print(
    "Kết quả:",
    ket_qua_du_doan.upper()
)


# ============================================================
# PHẦN 2: ID3 - DỰ ĐOÁN RỦI RO TÍN DỤNG
# ============================================================

credit_data = {

    'MaKH': [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    ],

    'Tuoi': [
        25, 40, 35, 27, 31,
        36, 48, 26, 33, 29,
        38, 44, 42, 28, 30
    ],

    'HonNhan': [
        'Độc thân',
        'Đã kết hôn',
        'Từng ly hôn',
        'Đã kết hôn',
        'Độc thân',
        'Đã kết hôn',
        'Độc thân',
        'Đã kết hôn',
        'Từng ly hôn',
        'Độc thân',
        'Đã kết hôn',
        'Độc thân',
        'Đã kết hôn',
        'Độc thân',
        'Đã kết hôn'
    ],

    'BatDongSan': [
        'Ở cùng bố mẹ',
        'Nhà sở hữu',
        'Nhà thuê',
        'Ở cùng bố mẹ',
        'Nhà thuê',
        'Nhà sở hữu',
        'Nhà thuê',
        'Nhà thuê',
        'Ở cùng bố mẹ',
        'Nhà thuê',
        'Nhà sở hữu',
        'Nhà sở hữu',
        'Nhà sở hữu',
        'Nhà thuê',
        'Ở cùng bố mẹ'
    ],

    'ThuNhap': [
        7000000,
        18000000,
        12000000,
        9000000,
        6000000,
        8000000,
        7000000,
        8000000,
        5000000,
        10000000,
        15000000,
        14000000,
        10000000,
        7000000,
        6000000
    ],

    'RuiRo': [
        0, 0, 1, 1, 1,
        1, 0, 1, 1, 0,
        0, 1, 0, 1, 1
    ]
}

bang_tin_dung = pd.DataFrame(
    credit_data
).set_index('MaKH')

print("\n\n========== DỮ LIỆU TÍN DỤNG ==========")

print(bang_tin_dung)


muc_tieu_tin_dung = 'RuiRo'


# Tìm ngưỡng tốt nhất cho thuộc tính số
def tim_nguong_tot_nhat(
    data_frame,
    thuoc_tinh,
    muc_tieu='RuiRo'
):

    entropy_goc = tinh_entropy(
        data_frame[muc_tieu]
    )

    cac_gia_tri = np.sort(
        data_frame[thuoc_tinh].unique()
    )

    if len(cac_gia_tri) <= 1:

        return 0, None, entropy_goc

    cac_nguong = (
        cac_gia_tri[:-1]
        + cac_gia_tri[1:]
    ) / 2.0

    gain_max = -1
    nguong_max = None
    entropy_max = entropy_goc

    for nguong in cac_nguong:

        tap_trai = data_frame[
            data_frame[thuoc_tinh] <= nguong
        ]

        tap_phai = data_frame[
            data_frame[thuoc_tinh] > nguong
        ]

        entropy_chia = (
            len(tap_trai) / len(data_frame)
        ) * tinh_entropy(
            tap_trai[muc_tieu]
        ) + (
            len(tap_phai) / len(data_frame)
        ) * tinh_entropy(
            tap_phai[muc_tieu]
        )

        gain = (
            entropy_goc
            - entropy_chia
        )

        if gain > gain_max:

            gain_max = gain
            nguong_max = nguong
            entropy_max = entropy_chia

    return (
        gain_max,
        nguong_max,
        entropy_max
    )


# Đánh giá các thuộc tính
def danh_gia_thuoc_tinh(
    data_frame,
    danh_sach,
    muc_tieu='RuiRo'
):

    ket_qua = {}

    for cot in danh_sach:

        if np.issubdtype(
            data_frame[cot].dtype,
            np.number
        ):

            gain, nguong, h_value = (
                tim_nguong_tot_nhat(
                    data_frame,
                    cot,
                    muc_tieu
                )
            )

            ket_qua[cot] = {
                'gain': gain,
                'threshold': nguong,
                'entropy': h_value,
                'type': 'number'
            }

        else:

            gia_tri, so_luong = np.unique(
                data_frame[cot],
                return_counts=True
            )

            entropy_chia = 0

            for i in range(
                len(gia_tri)
            ):

                tap_con = data_frame[
                    data_frame[cot]
                    == gia_tri[i]
                ]

                entropy_chia += (
                    so_luong[i]
                    / len(data_frame)
                ) * tinh_entropy(
                    tap_con[muc_tieu]
                )

            gain = (
                tinh_entropy(
                    data_frame[muc_tieu]
                )
                - entropy_chia
            )

            ket_qua[cot] = {
                'gain': gain,
                'threshold': None,
                'entropy': entropy_chia,
                'type': 'category'
            }

    return ket_qua


cac_thuoc_tinh_tin_dung = [
    'Tuoi',
    'HonNhan',
    'BatDongSan',
    'ThuNhap'
]


print("\n========== ID3 TÍN DỤNG ==========")

print(
    f"Entropy ban đầu = "
    f"{tinh_entropy(bang_tin_dung[muc_tieu_tin_dung]):.4f}"
)


ket_qua_danh_gia = danh_gia_thuoc_tinh(
    bang_tin_dung,
    cac_thuoc_tinh_tin_dung,
    muc_tieu_tin_dung
)

bang_id3_tin_dung = []

for ten, thong_tin in ket_qua_danh_gia.items():

    if thong_tin['type'] == 'number':

        dieu_kien = (
            f"<= {thong_tin['threshold']}"
        )

    else:

        dieu_kien = "Theo nhóm"

    bang_id3_tin_dung.append({

        'Thuoc tinh': ten,

        'Dieu kien': dieu_kien,

        'H(X,S)': round(
            thong_tin['entropy'],
            4
        ),

        'Information Gain': round(
            thong_tin['gain'],
            4
        )
    })


print(
    pd.DataFrame(
        bang_id3_tin_dung
    ).sort_values(
        by='Information Gain',
        ascending=False
    )
)


# Xây dựng cây ID3 cho tín dụng
def tao_cay_tin_dung(
    du_lieu,
    du_lieu_goc,
    danh_sach,
    muc_tieu='RuiRo',
    lop_cha=None
):

    if len(du_lieu) == 0:

        gia_tri, so_luong = np.unique(
            du_lieu_goc[muc_tieu],
            return_counts=True
        )

        return gia_tri[
            np.argmax(so_luong)
        ]

    if len(np.unique(
        du_lieu[muc_tieu]
    )) <= 1:

        return np.unique(
            du_lieu[muc_tieu]
        )[0]

    if len(danh_sach) == 0:

        return lop_cha

    gia_tri, so_luong = np.unique(
        du_lieu[muc_tieu],
        return_counts=True
    )

    lop_cha = gia_tri[
        np.argmax(so_luong)
    ]

    danh_gia = danh_gia_thuoc_tinh(
        du_lieu,
        danh_sach,
        muc_tieu
    )

    thuoc_tinh_chon = max(
        danh_gia,
        key=lambda x:
        danh_gia[x]['gain']
    )

    thong_tin = danh_gia[
        thuoc_tinh_chon
    ]

    if thong_tin['gain'] <= 1e-6:

        return lop_cha

    cay = {}

    if thong_tin['type'] == 'category':

        cay[thuoc_tinh_chon] = {}

        danh_sach_moi = [
            x for x in danh_sach
            if x != thuoc_tinh_chon
        ]

        for gia_tri in np.unique(
            du_lieu[thuoc_tinh_chon]
        ):

            tap_con = du_lieu[
                du_lieu[thuoc_tinh_chon]
                == gia_tri
            ]

            cay[
                thuoc_tinh_chon
            ][gia_tri] = tao_cay_tin_dung(
                tap_con,
                du_lieu_goc,
                danh_sach_moi,
                muc_tieu,
                lop_cha
            )

    else:

        nguong = thong_tin[
            'threshold'
        ]

        ten_nut = (
            f"{thuoc_tinh_chon} <= {nguong:g}"
        )

        cay[ten_nut] = {}

        tap_trai = du_lieu[
            du_lieu[thuoc_tinh_chon]
            <= nguong
        ]

        tap_phai = du_lieu[
            du_lieu[thuoc_tinh_chon]
            > nguong
        ]

        cay[ten_nut]['Yes'] = (
            tao_cay_tin_dung(
                tap_trai,
                du_lieu_goc,
                danh_sach,
                muc_tieu,
                lop_cha
            )
        )

        cay[ten_nut]['No'] = (
            tao_cay_tin_dung(
                tap_phai,
                du_lieu_goc,
                danh_sach,
                muc_tieu,
                lop_cha
            )
        )

    return cay


def in_cay_tin_dung(
    cay,
    khoang_cach=""
):

    if not isinstance(cay, dict):

        print(
            f" ──> [Rủi ro = {cay}]"
        )

        return

    for nut, nhanh in cay.items():

        for gia_tri, cay_con in nhanh.items():

            print(
                f"{khoang_cach}├── "
                f"({nut} == '{gia_tri}')",
                end=""
            )

            if isinstance(cay_con, dict):

                print()

                in_cay_tin_dung(
                    cay_con,
                    khoang_cach + "│   "
                )

            else:

                print(
                    f" ──> [Rủi ro = {cay_con}]"
                )


cay_tin_dung = tao_cay_tin_dung(
    bang_tin_dung,
    bang_tin_dung,
    cac_thuoc_tinh_tin_dung,
    muc_tieu_tin_dung
)

print(
    "\n========== CÂY ID3 TÍN DỤNG =========="
)

in_cay_tin_dung(
    cay_tin_dung
)


# Dự đoán bằng ID3
def du_doan_tin_dung(
    cay,
    thong_tin_khach_hang
):

    if not isinstance(cay, dict):

        return cay

    nut = list(cay.keys())[0]

    if "<=" in nut:

        ten_thuoc_tinh, gia_tri = (
            nut.split(" <= ")
        )

        nguong = float(gia_tri)

        gia_tri_khach = (
            thong_tin_khach_hang[
                ten_thuoc_tinh
            ]
        )

        if gia_tri_khach <= nguong:

            nhanh = "Yes"

        else:

            nhanh = "No"

        return du_doan_tin_dung(
            cay[nut][nhanh],
            thong_tin_khach_hang
        )

    else:

        gia_tri = (
            thong_tin_khach_hang[nut]
        )

        if gia_tri in cay[nut]:

            return du_doan_tin_dung(
                cay[nut][gia_tri],
                thong_tin_khach_hang
            )

        return "Không có nhánh"


khach_hang_1 = {

    'Tuoi': 32,

    'HonNhan': 'Đã kết hôn',

    'BatDongSan': 'Nhà thuê',

    'ThuNhap': 12000000
}


ket_qua_tin_dung = du_doan_tin_dung(
    cay_tin_dung,
    khach_hang_1
)

print(
    "\n========== DỰ ĐOÁN ID3 =========="
)

print(
    "Khách hàng:",
    khach_hang_1
)

print(
    "Rủi ro tín dụng:",
    ket_qua_tin_dung
)


# ============================================================
# PHẦN 3: CART - GINI INDEX
# ============================================================

# Hàm tính Gini
def tinh_gini(cot_muc_tieu):

    if len(cot_muc_tieu) == 0:

        return 0

    _, so_luong = np.unique(
        cot_muc_tieu,
        return_counts=True
    )

    xac_suat = (
        so_luong
        / len(cot_muc_tieu)
    )

    return 1 - np.sum(
        xac_suat ** 2
    )


# Tìm phép chia CART tốt nhất
def tim_chia_cart(
    data_frame,
    thuoc_tinh,
    muc_tieu='RuiRo'
):

    tong_so = len(data_frame)

    gini_tot_nhat = 1.0

    dieu_kien_tot_nhat = None

    tap_tot_nhat = None

    # Thuộc tính số
    if np.issubdtype(
        data_frame[thuoc_tinh].dtype,
        np.number
    ):

        gia_tri = np.sort(
            data_frame[
                thuoc_tinh
            ].unique()
        )

        if len(gia_tri) <= 1:

            return (
                1.0,
                None,
                None
            )

        cac_nguong = (
            gia_tri[:-1]
            + gia_tri[1:]
        ) / 2

        for nguong in cac_nguong:

            trai = data_frame[
                data_frame[thuoc_tinh]
                <= nguong
            ]

            phai = data_frame[
                data_frame[thuoc_tinh]
                > nguong
            ]

            gini_chia = (
                len(trai) / tong_so
            ) * tinh_gini(
                trai[muc_tieu]
            ) + (
                len(phai) / tong_so
            ) * tinh_gini(
                phai[muc_tieu]
            )

            if gini_chia < gini_tot_nhat:

                gini_tot_nhat = gini_chia

                dieu_kien_tot_nhat = (
                    'number',
                    nguong
                )

                tap_tot_nhat = (
                    trai,
                    phai
                )

    # Thuộc tính phân loại
    else:

        cac_nhom = list(
            data_frame[
                thuoc_tinh
            ].unique()
        )

        if len(cac_nhom) <= 1:

            return (
                1.0,
                None,
                None
            )

        for kich_thuoc in range(
            1,
            len(cac_nhom) // 2 + 1
        ):

            for nhom in combinations(
                cac_nhom,
                kich_thuoc
            ):

                nhom = list(nhom)

                trai = data_frame[
                    data_frame[
                        thuoc_tinh
                    ].isin(nhom)
                ]

                phai = data_frame[
                    ~data_frame[
                        thuoc_tinh
                    ].isin(nhom)
                ]

                gini_chia = (
                    len(trai) / tong_so
                ) * tinh_gini(
                    trai[muc_tieu]
                ) + (
                    len(phai) / tong_so
                ) * tinh_gini(
                    phai[muc_tieu]
                )

                if gini_chia < gini_tot_nhat:

                    gini_tot_nhat = gini_chia

                    dieu_kien_tot_nhat = (
                        'category',
                        nhom
                    )

                    tap_tot_nhat = (
                        trai,
                        phai
                    )

    return (
        gini_tot_nhat,
        dieu_kien_tot_nhat,
        tap_tot_nhat
    )


print(
    "\n\n========== CART - GINI =========="
)

print(
    "Gini nút gốc:",
    round(
        tinh_gini(
            bang_tin_dung[
                muc_tieu_tin_dung
            ]
        ),
        4
    )
)


bang_cart = []

for thuoc_tinh in cac_thuoc_tinh_tin_dung:

    gini_value, dieu_kien, _ = (
        tim_chia_cart(
            bang_tin_dung,
            thuoc_tinh,
            muc_tieu_tin_dung
        )
    )

    if dieu_kien:

        if dieu_kien[0] == 'number':

            mo_ta = (
                f"<= {dieu_kien[1]:g}"
            )

        else:

            mo_ta = (
                f"thuộc {dieu_kien[1]}"
            )

    else:

        mo_ta = "None"

    bang_cart.append({

        'Thuoc tinh': thuoc_tinh,

        'Dieu kien': mo_ta,

        'Gini Split': round(
            gini_value,
            4
        )
    })


print(
    pd.DataFrame(
        bang_cart
    ).sort_values(
        by='Gini Split'
    )
)


# Xây dựng cây CART
def tao_cay_cart(
    data_frame,
    du_lieu_goc,
    danh_sach,
    muc_tieu='RuiRo',
    do_sau_toi_da=5,
    do_sau_hien_tai=0
):

    if len(data_frame) == 0:

        gia_tri, so_luong = np.unique(
            du_lieu_goc[muc_tieu],
            return_counts=True
        )

        return gia_tri[
            np.argmax(so_luong)
        ]

    if (
        tinh_gini(
            data_frame[muc_tieu]
        ) == 0
        or
        do_sau_hien_tai >= do_sau_toi_da
    ):

        gia_tri, so_luong = np.unique(
            data_frame[muc_tieu],
            return_counts=True
        )

        return gia_tri[
            np.argmax(so_luong)
        ]

    gini_tot_nhat = 1.0

    thuoc_tinh_tot_nhat = None

    dieu_kien_tot_nhat = None

    tap_tot_nhat = None

    for thuoc_tinh in danh_sach:

        gini_value, dieu_kien, cac_tap = (
            tim_chia_cart(
                data_frame,
                thuoc_tinh,
                muc_tieu
            )
        )

        if (
            cac_tap is not None
            and gini_value < gini_tot_nhat
        ):

            gini_tot_nhat = gini_value

            thuoc_tinh_tot_nhat = thuoc_tinh

            dieu_kien_tot_nhat = dieu_kien

            tap_tot_nhat = cac_tap

    gini_hien_tai = tinh_gini(
        data_frame[muc_tieu]
    )

    if (
        thuoc_tinh_tot_nhat is None
        or gini_tot_nhat >= gini_hien_tai
    ):

        gia_tri, so_luong = np.unique(
            data_frame[muc_tieu],
            return_counts=True
        )

        return gia_tri[
            np.argmax(so_luong)
        ]

    tap_trai, tap_phai = tap_tot_nhat

    cay = {}

    if dieu_kien_tot_nhat[0] == 'number':

        nguong = dieu_kien_tot_nhat[1]

        luat_trai = (
            f"{thuoc_tinh_tot_nhat} <= {nguong:g}"
        )

        luat_phai = (
            f"{thuoc_tinh_tot_nhat} > {nguong:g}"
        )

    else:

        nhom = dieu_kien_tot_nhat[1]

        luat_trai = (
            f"{thuoc_tinh_tot_nhat} in {nhom}"
        )

        luat_phai = (
            f"{thuoc_tinh_tot_nhat} not in {nhom}"
        )

    cay[f"Kiem tra: {thuoc_tinh_tot_nhat}"] = {

        luat_trai: tao_cay_cart(
            tap_trai,
            du_lieu_goc,
            danh_sach,
            muc_tieu,
            do_sau_toi_da,
            do_sau_hien_tai + 1
        ),

        luat_phai: tao_cay_cart(
            tap_phai,
            du_lieu_goc,
            danh_sach,
            muc_tieu,
            do_sau_toi_da,
            do_sau_hien_tai + 1
        )
    }

    return cay


def in_cay_cart(
    cay,
    khoang_cach=""
):

    if not isinstance(cay, dict):

        print(
            f" ──> [Rủi ro tín dụng = {cay}]"
        )

        return

    for nut, nhanh in cay.items():

        for dieu_kien, cay_con in nhanh.items():

            print(
                f"{khoang_cach}├── "
                f"IF ({dieu_kien})",
                end=""
            )

            if isinstance(cay_con, dict):

                print()

                in_cay_cart(
                    cay_con,
                    khoang_cach + "│   "
                )

            else:

                print(
                    f" ──> "
                    f"[Rủi ro tín dụng = {cay_con}]"
                )


cay_cart = tao_cay_cart(
    bang_tin_dung,
    bang_tin_dung,
    cac_thuoc_tinh_tin_dung,
    muc_tieu_tin_dung
)

print(
    "\n========== CÂY QUYẾT ĐỊNH CART =========="
)

in_cay_cart(
    cay_cart
)


# Hàm dự đoán CART
def du_doan_cart(
    cay,
    thong_tin
):

    if not isinstance(cay, dict):

        return cay

    nut = list(cay.keys())[0]

    cac_nhanh = cay[nut]

    for dieu_kien, cay_con in cac_nhanh.items():

        if " <= " in dieu_kien:

            ten_cot, gia_tri = (
                dieu_kien.split(" <= ")
            )

            if (
                thong_tin[ten_cot]
                <= float(gia_tri)
            ):

                return du_doan_cart(
                    cay_con,
                    thong_tin
                )

        elif " > " in dieu_kien:

            ten_cot, gia_tri = (
                dieu_kien.split(" > ")
            )

            if (
                thong_tin[ten_cot]
                > float(gia_tri)
            ):

                return du_doan_cart(
                    cay_con,
                    thong_tin
                )

        elif " not in " in dieu_kien:

            ten_cot, danh_sach = (
                dieu_kien.split(" not in ")
            )

            nhom = eval(danh_sach)

            if thong_tin[ten_cot] not in nhom:

                return du_doan_cart(
                    cay_con,
                    thong_tin
                )

        elif " in " in dieu_kien:

            ten_cot, danh_sach = (
                dieu_kien.split(" in ")
            )

            nhom = eval(danh_sach)

            if thong_tin[ten_cot] in nhom:

                return du_doan_cart(
                    cay_con,
                    thong_tin
                )

    return "Không xác định"


# Khách hàng kiểm tra CART
khach_hang_cart = {

    'Tuoi': 35,

    'HonNhan': 'Đã kết hôn',

    'BatDongSan': 'Nhà sở hữu',

    'ThuNhap': 12000000
}


ket_qua_cart = du_doan_cart(
    cay_cart,
    khach_hang_cart
)

print(
    "\n========== DỰ ĐOÁN CART =========="
)

print(
    "Khách hàng:",
    khach_hang_cart
)

print(
    "Dự đoán rủi ro tín dụng:",
    ket_qua_cart
)