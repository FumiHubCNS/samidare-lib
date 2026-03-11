import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    from pyspark.sql import Row
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F, types as T
    from pyspark.sql import Window
    from pyspark.sql.functions import pandas_udf

    from math import isfinite
    from typing import Sequence, Tuple, Dict, Any

    import catm_lib as cat
    import catm_lib.util.catmviewer as catview
    import samidare_lib.util.padinfo as padinfo

    import pandas as pd
    import samidare_lib.core as dev
    import samidare_lib.util as util
    import marimo_lib.util as molib


@app.function
def calculate_energy_deposit(val, a=0.2327, b=2.375, Cg=4.0, Cmv=1.953, W=26e-3, e=1.602e-4, Gg=2781.):
    Qch   = val * a + b
    Qmeas = Qch * Cmv / Cg
    dE = Qmeas / (( e / W ) * Gg )
    return dE


@app.function
def calculate_energy_gain(val, a=0.2327, b=2.375, Cg=4.0, Cmv=1.953, W=26e-3, e=1.602e-4, dE=133.):
    Qch   = val * a + b
    Qmeas = Qch * Cmv / Cg
    Gg = Qmeas / (( e / W ) * dE )

    return Gg


@app.function
def calculate_energy_conversion_factor(a=0.2327, b=2.375, Cg=4.0, Cmv=1.953, W=26e-3, e=1.602e-4, Gg=2781.):
    coeff0 = a*Cmv*W/(Cg*e*Gg) 
    coeff1 = b*Cmv*W/(Cg*e*Gg) 
    return [coeff0, coeff1]


@app.cell
def _(fit_global_odr, np, plt):
    def check_fit_alg(x,y,q):

        chi2, res = fit_global_odr(x, y, q)
        print("chi2 =", chi2)
        print("angle(deg) =", res["angle_deg"])
        print("line: y = {:.6g} x + {}".format(res["slope"], res["intercept"]))


        # 数値結果を表に
        df = pd.DataFrame([{
            "chi2": chi2,
            "chi2_reduced": res["chi2_reduced"],
            "slope": res["slope"],
            "intercept": res["intercept"],
            "angle_deg": res["angle_deg"],
            "n_points": res["n_points"],
            "dof": res["dof"],
            "centroid_x": res["centroid"][0],
            "centroid_y": res["centroid"][1],
        }])

        # 可視化（データ点とフィット直線）
        X = np.asarray(x, dtype=float)
        Y = np.asarray(y, dtype=float)
        nx, ny, c = res["normal"]

        xmin, xmax = X.min(), X.max()
        ymin, ymax = Y.min(), Y.max()
        xm = (xmax - xmin) if xmax > xmin else 1.0
        ym = (ymax - ymin) if ymax > ymin else 1.0
        padx = 0.1 * xm
        pady = 0.1 * ym

        fig = plt.figure(figsize=(6, 4.5))

        # 散布図（重みはサイズに反映）
        sizes = (np.asarray(q) / np.max(q)) * 80 + 20  # 20〜100あたり
        plt.scatter(X, Y, s=sizes, label="data (size ~ weight)")

        # フィット直線の描画
        if np.isfinite(res["slope"]):
            xs = np.linspace(xmin - padx, xmax + padx, 200)
            ys = res["slope"] * xs + (res["intercept"] if res["intercept"] is not None else 0.0)
            plt.plot(xs, ys, label="fit line")
        else:
            # 垂直直線 x = -c/nx
            x0 = -c / nx
            ys = np.linspace(ymin - pady, ymax + pady, 200)
            plt.plot([x0, x0], [ys.min(), ys.max()], label="fit line (vertical)")

        plt.xlabel("x")
        plt.ylabel("y")
        title = f"Global ODR fit  (χ² = {chi2:.4g}"
        if res["chi2_reduced"] is not None:
            title += f",  χ²/ν = {res['chi2_reduced']:.4g}"
        title += f",  angle = {res['angle_deg']:.3g}°)"
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return


@app.cell
def _(np):
    def fit_line_ols(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
        """
        y = a x + b を最小二乗でフィットする。
        返り値: dict
          - slope: a
          - intercept: b
          - r2: 決定係数
          - rss: 残差二乗和 (＝χ² 相当、誤差分散未知のとき)
          - dof: 自由度 (n-2)
          - stderr_slope: a の標準誤差
          - stderr_intercept: b の標準誤差
          - cov_ab: a,b の共分散
          - x_mean, y_mean: 平均
          - predict(x_new): 予測関数（ベクトル/スカラー対応）
        """
        X = np.asarray(x, dtype=float)
        Y = np.asarray(y, dtype=float)
        if X.shape != Y.shape:
            raise ValueError("x と y の長さは一致している必要があります。")

        # 有効値のみ使用
        m = np.isfinite(X) & np.isfinite(Y)
        X = X[m]; Y = Y[m]
        n = X.size
        if n < 2:
            raise ValueError("少なくとも2点が必要です。")

        x_mean = X.mean()
        y_mean = Y.mean()
        dx = X - x_mean
        dy = Y - y_mean
        Sxx = np.sum(dx * dx)
        if Sxx == 0.0:
            raise ValueError("x が全て同一です。y = a x + b ではフィットできません。")
        Sxy = np.sum(dx * dy)

        # 推定量
        a = Sxy / Sxx
        b = y_mean - a * x_mean

        # 残差と指標
        yhat = a * X + b
        resid = Y - yhat
        rss = float(np.sum(resid * resid))
        tss = float(np.sum((Y - y_mean) ** 2))
        r2 = 1.0 - rss / tss if tss > 0 else (1.0 if rss < 1e-30 else 0.0)

        dof = max(n - 2, 0)
        if dof > 0:
            s2 = rss / dof
            stderr_a = float(np.sqrt(s2 / Sxx))
            stderr_b = float(np.sqrt(s2 * (1.0 / n + (x_mean ** 2) / Sxx)))
            cov_ab = float(-x_mean * s2 / Sxx)
        else:
            stderr_a = float("nan")
            stderr_b = float("nan")
            cov_ab = float("nan")

        def predict(x_new):
            xn = np.asarray(x_new, dtype=float)
            return a * xn + b

        return {
            "slope": float(a),
            "intercept": float(b),
            "r2": float(r2),
            "rss": float(rss),
            "dof": int(dof),
            "stderr_slope": stderr_a,
            "stderr_intercept": stderr_b,
            "cov_ab": cov_ab,
            "x_mean": float(x_mean),
            "y_mean": float(y_mean),
            "predict": predict,
        }

    return (fit_line_ols,)


@app.cell
def _(np):
    def fit_global_odr(x: Sequence[float],
                       y: Sequence[float],
                       q: Sequence[float],
                       *,
                       return_reduced_chi2: bool = True
                       ) -> Tuple[float, Dict[str, Any]]:
        """
        Weighted Orthogonal Distance Regression (Global Fitting).
        - 入力: x, y, q（同じ長さ）
          q は各点の重み（例: 誘起電荷）。負や0、非有限は除外。
        - 目的関数: χ² = Σ_i q_i * d_i^2
          ここで d_i は点 (x_i, y_i) と直線の「直交距離」。
        - 出力:
            chi2: 上式の χ²
            result(dict):
              - slope: 直線 y = a x + b の a（ほぼ垂直なら np.inf）
              - intercept: b（ほぼ垂直なら None）
              - normal: (nx, ny, c) 単位法線での行列表現 nx*x + ny*y + c = 0
              - angle_rad / angle_deg: 直線方向ベクトルの角度（x軸に対して）
              - centroid: (cx, cy) 加重重心
              - n_points: 使用点数
              - dof: 自由度（= n_points - 2、<0なら 0）
              - chi2_reduced: χ²/DOF（計算可能な場合のみ）
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(q, dtype=float)

        if x.shape != y.shape or x.shape != w.shape:
            raise ValueError("x, y, q の長さは同じである必要があります。")

        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        if not np.any(mask):
            raise ValueError("有効なデータ点がありません（非有限/重み<=0 しかない）。")

        X = x[mask]
        Y = y[mask]
        W = w[mask]
        n_points = X.size
        if n_points < 2:
            raise ValueError("フィットには少なくとも2点が必要です。")

        # --- 加重重心
        Wsum = W.sum()
        cx = (W * X).sum() / Wsum
        cy = (W * Y).sum() / Wsum

        # --- 加重共分散行列（PCAで主軸=直線方向を得る）
        dx = X - cx
        dy = Y - cy
        Sxx = (W * dx * dx).sum() / Wsum
        Syy = (W * dy * dy).sum() / Wsum
        Sxy = (W * dx * dy).sum() / Wsum
        S = np.array([[Sxx, Sxy],
                      [Sxy, Syy]], dtype=float)

        # 固有分解（対象行列）
        vals, vecs = np.linalg.eigh(S)  # 昇順の固有値が返る
        d = vecs[:, np.argmax(vals)]    # 最大固有値に対応する固有ベクトル = 直線方向
        d = d / np.linalg.norm(d)

        # 法線ベクトル（単位）と c： nx*x + ny*y + c = 0
        n = np.array([-d[1], d[0]], dtype=float)  # 方向ベクトルに直交
        n = n / np.linalg.norm(n)
        c = - (n[0] * cx + n[1] * cy)

        # --- χ²（直交距離の重み付き二乗和）を計算
        di = n[0] * X + n[1] * Y + c          # 単位法線なので |di| が直交距離
        chi2 = float((W * di**2).sum())

        # y = a x + b 形式（ほぼ垂直の扱い）
        eps = 1e-12
        if abs(n[1]) < eps:
            slope = float('inf')
            intercept = None
        else:
            slope = - n[0] / n[1]
            intercept = - c / n[1]

        # 角度（直線方向ベクトル d の角度）
        angle_rad = float(np.arctan2(d[1], d[0]))
        angle_deg = float(np.degrees(angle_rad))

        dof = max(n_points - 2, 0)
        chi2_reduced = (chi2 / dof) if (return_reduced_chi2 and dof > 0) else None

        result = {
            "slope": slope,
            "intercept": intercept,
            "normal": (float(n[0]), float(n[1]), float(c)),
            "angle_rad": angle_rad,
            "angle_deg": angle_deg,
            "centroid": (float(cx), float(cy)),
            "n_points": int(n_points),
            "dof": int(dof),
            "chi2_reduced": None if chi2_reduced is None else float(chi2_reduced),
        }
        return chi2, result

    odr_schema = T.StructType([
        T.StructField("event_id",      T.LongType()),
        T.StructField("a_slope",       T.DoubleType()),
        T.StructField("a_intercept",   T.DoubleType()),
        T.StructField("a_chi2",        T.DoubleType()),
        T.StructField("a_chi2_reduced",T.DoubleType()),
        T.StructField("a_angle_deg",   T.DoubleType()),
        T.StructField("a_n_points",    T.IntegerType()),
        T.StructField("a_dof",         T.IntegerType()),
    ])
    return fit_global_odr, odr_schema


@app.cell
def _(fit_global_odr, np):
    def odr_group(pdf: pd.DataFrame) -> pd.DataFrame:
        eid = int(pdf["event_id"].iloc[0])
        X = pdf["x"].to_numpy(float)
        Z = pdf["z"].to_numpy(float)
        W = pdf["w"].to_numpy(float)

        out = {"event_id": eid, "a_slope": np.nan, "a_intercept": np.nan,
               "a_chi2": np.nan, "a_chi2_reduced": np.nan, "a_angle_deg": np.nan,
               "a_n_points": int(len(X)), "a_dof": max(int(len(X))-2, 0)}
        try:
            if len(X) >= 2 and np.isfinite(X).all() and np.isfinite(Z).all() and np.isfinite(W).all():
                chi2, res = fit_global_odr(X, Z, W)
                out.update({
                    "a_slope": float(res["slope"]) if np.isfinite(res["slope"]) else np.nan,
                    "a_intercept": (None if res["intercept"] is None else float(res["intercept"])),
                    "a_chi2": float(chi2),
                    "a_chi2_reduced": (None if res["chi2_reduced"] is None else float(res["chi2_reduced"])),
                    "a_angle_deg": float(res["angle_deg"]),
                    "a_n_points": int(res["n_points"]),
                    "a_dof": int(res["dof"]),
                })
        except Exception:
            pass
        return pd.DataFrame([out])

    return (odr_group,)


@app.function
def within_range(arr, low, high, *, inclusive="both", abs_value=False, finite_only=True):
    """
    arr中の値のうち [low, high]（inclusive='both' のとき）に入る要素だけ返す。
      inclusive: 'both' | 'left' | 'right' | 'neither'
      abs_value: True で |x| を範囲判定に使う
      finite_only: True で inf/-inf/NaN/None を除外
    """
    if low > high:
        low, high = high, low

    def ok(x):
        if not isinstance(x, (int, float)):
            return False
        if finite_only and not isfinite(x):
            return False
        v = abs(x) if abs_value else x
        if   inclusive == "both":    return low <= v <= high
        elif inclusive == "left":    return low <= v <  high
        elif inclusive == "right":   return low <  v <= high
        elif inclusive == "neither": return low <  v <  high
        else:  # 不正値は既定で両端含む
            return low <= v <= high

    return [x for x in arr if ok(x)]


@app.function
def within_range_indices(arr, low, high, **kwargs):
    """範囲内に入る要素のインデックスを返す版"""
    vals = within_range(arr, low, high, **kwargs)
    # 同じ条件で再判定して index を返す（重複回避なら一度条件関数を外に出す実装でもOK）
    res = []
    if low > high:
        low, high = high, low
    abs_value = kwargs.get("abs_value", False)
    inclusive = kwargs.get("inclusive", "both")
    finite_only = kwargs.get("finite_only", True)

    def cond(x):
        if not isinstance(x, (int, float)): return False
        if finite_only and not isfinite(x): return False
        v = abs(x) if abs_value else x
        if   inclusive == "both":    return low <= v <= high
        elif inclusive == "left":    return low <= v <  high
        elif inclusive == "right":   return low <  v <= high
        elif inclusive == "neither": return low <  v <  high
        return low <= v <= high

    return [i for i, x in enumerate(arr) if cond(x)]


@app.function
def get_id_from_mapdf(mapdf, sampaNo=2, sampaID=4, label='gid'):
    matched = mapdf.loc[(mapdf['sampaNo'] == sampaNo) & (mapdf['sampaID'] == sampaID), label]
    gid = matched.iloc[0] if not matched.empty else None
    return gid


@app.function
def get_any_from_mapdf_using_ref(mapdf,refLabel='samidareID', refID=4, label='gid'):
    matched = mapdf.loc[(mapdf[refLabel] == refID), label]
    gid = matched.iloc[0] if not matched.empty else None
    return gid


@app.function
def get_any_from_mapdf(mapdf, refLabel='sampaNo', refIDID=4):
    matched = mapdf[(mapdf[refLabel] == refIDID)]
    return matched


@app.function
def get_pads():
    offset = -3.031088913245535
    pad1 = padinfo.get_tpc_info(offset+45)
    pad2 = padinfo.get_tpc_info(offset+136.5,False)
    tpcs = padinfo.marge_padinfos(pad1,pad2)
    tpcs.dedxinfo = {
        "Am241": { 
            "tke": 5.5,
            "up":   { "de[keV]": 133.3, "sigma_de[keV/u]": 0.002, "sigma_angle[mrad]": 7.3441},
            "down": { "de[keV]": 175.7, "sigma_de[keV/u]": 0.002, "sigma_angle[mrad]": 11.733},
            },
        "Cm244": { 
            "tke": 5.8,
            "up":   { "de[keV]": 129.8, "sigma_de[keV/u]": 0.002, "sigma_angle[mrad]": 6.8752},
            "down": { "de[keV]": 162.9, "sigma_de[keV/u]": 0.002, "sigma_angle[mrad]": 10.284},
            },
        "Np237": { 
            "tke": 4.959,
            "up":   { "de[keV]": 145.6, "sigma_de[keV/u]": 0.002, "sigma_angle[mrad]":  8.3864},
            "down": { "de[keV]": 209.9, "sigma_de[keV/u]": 0.002, "sigma_angle[mrad]": 16.1315},
            },
    }

    return tpcs


@app.function
def calculate_position_charge(
        pad:cat.basepad.TReadoutPadArray = None, 
        vids:list = None, 
        tpcid:list =None,
        charge:list = None,
        time:list = None
        ): 
    ps = []
    qs= []

    for idx in vids:    
        id = padinfo.find_index(pad.ids, int(tpcid[int(idx)]))
        pos = pad.centers[id]
        pos[1] = time[idx]
        ps.append(pos)
        qs.append(charge[idx])

    return (ps, qs)


@app.function
def calculate_weighted_average(vals, wts):
    return sum(v*w for v, w in zip(vals, wts)) / sum(wts)


@app.cell
def _(make_subplots, pau, saveutil):
    def check_tpc_map(data=None, input_finename="", savebase=None, save_flag=False):
        if data is None:
            print(f"input data is none. return None")
            return None

        data1 = (data.groupBy("samidare_id").count().orderBy("samidare_id"))
        data2 = (data.groupBy("tpc_id").count().orderBy("tpc_id"))

        df1 = data1.toPandas()
        df2 = data2.toPandas()

        fig = make_subplots(rows=1, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1,
            subplot_titles=( "Samidare ID", "TPC ID" ))

        pau.add_sub_plot(fig,1,1,'spark-hist',[df1["samidare_id"],df1["count"]],['Samidare ID','Counts'])
        pau.add_sub_plot(fig,1,2,'spark-hist',[df2["tpc_id"],df2["count"]],['TPC ID','Counts'])

        fig.update_layout( height=800, width=1600, showlegend=False,title_text=f"{input_finename}")
        fig.show()

        if save_flag and savebase is not None:
            saveutil.save_plotly(fig, base_dir=savebase)

    return (check_tpc_map,)


@app.cell
def _(fit_line_ols, make_subplots, np, pau, saveutil):
    def check_hq(df1=None, df2=None, input_finename="", savebase=None, save_flag=False):
        data1 = df1.toPandas()
        data2 = df2.toPandas()
        res = fit_line_ols(data1["charge"],data1["maxsample"])
        print(f'a = {res["slope"]:.5f}, b = {res["intercept"]:.5f} (chrage < 4000)')
        fitx =  np.linspace(0,4000, 5)
        fity = _pol1(fitx,res['slope'],res["intercept"])


        fig = make_subplots(rows=1, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1,subplot_titles=("h x q (raw)",f"h x q (valid), a = {res['slope']:.3f}, b = {res['intercept']:.3f}",))
        pau.add_sub_plot(fig,1,1,'2d',[data2["charge"],data2["maxsample"]],["Charge [ch]"," Max Sample [ch]"],[500,500],yrange=[0,1100],xrange=[0,6000])
        pau.add_sub_plot(fig,1,2,'2d',[data1["charge"],data1["maxsample"]],["Charge [ch]"," Max Sample [ch]"],[500,500],yrange=[0,1100],xrange=[0,6000])
        pau.add_sub_plot(fig,1,2,'fit',[fitx, fity])
        fig.update_layout( height=700, width=1400, showlegend=False,title_text=f"{input_finename}")
        fig.show()

        if save_flag and savebase is not None:
            saveutil.save_plotly(fig, base_dir=savebase)

    return (check_hq,)


@app.cell
def _(make_subplots, pau, saveutil):
    def check_nq(data=None, input_finename="", savebase=None, save_flag=False):
        df_fit = data
        df_plot2 = df_fit.select("N_up","ΔE_up","N_dn","ΔE_dn")
        data1 = df_plot2.toPandas()

        fig = make_subplots(rows=1, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1,subplot_titles=("Nup vs dEsum,up","Ndown vs dEsum,down",))
        pau.add_sub_plot(fig,1,1,'2d',[data1["N_up"],data1["ΔE_up"]],[r'$N_{up}$',r'$\Delta E_{up}$ [keV]'],[25,200])
        pau.add_sub_plot(fig,1,2,'2d',[data1["N_dn"],data1["ΔE_dn"]],[r'$N_{down}$',r'$\Delta E_{down}$ [keV]'],[25,200])
        pau.align_colorbar(fig)
        fig.update_layout( height=700, width=1400, showlegend=False,title_text=f"{input_finename}")
        fig.show()

        if save_flag and savebase is not None:
            saveutil.save_plotly(fig, base_dir=savebase)

    return (check_nq,)


@app.function
def check_hit_patturn(data,block_flag=False,check_flag=True, savefilepath=None, dpi=300, maxevt=-1):

    tpcs = get_pads()

    figcount = 0

    for ev in data.toLocalIterator():
        N_up = ev["N_up"]
        N_dn = ev["N_dn"]
        slope = ev["a_slope"]
        intercept = ev["a_intercept"]

        if N_up > 3 and N_dn > 3 :

            if figcount == maxevt:
                break 

            tpcid_arr = list(ev["tpc_ids"] or [])
            de_arr    = list(ev["energy_deposits_keV"] or [])

            reflist = de_arr
            q_lst = [0] * len(tpcs.ids)

            for i in range(len(reflist)):
                id = padinfo.find_index(tpcs.ids, int(tpcid_arr[i]))
                q_lst[id] = int(reflist[i])

            cehck_list = q_lst
            bins, colors = catview.get_color_list(cehck_list, cmap_name="ocean_r", fmt="hex")
            color_array  = catview.get_color_array(cehck_list,bins,colors)

            tracks  = []
            tracks.append(["line",[1/slope, -intercept/slope], [40,141], [1,'red']])
            outputpath = f"{savefilepath}/{figcount:05d}.png" if savefilepath else None

            tpcs.show_pads(
                plot_type='map', 
                color_map=color_array, 
                xrange=[-20,20],
                yrange=[38,142],
                block_flag=block_flag,
                savepath = outputpath,
                check_id = check_flag,
                check_size=3,
                check_data = tpcs.ids,
                canvassize = [8,7],
                tracks=tracks,
                dpi=dpi
            )

            figcount += 1


@app.cell
def _(make_subplots, pau, saveutil):
    def check_fit_result(data, input_finename="", savebase=None, save_flag=False):
        data1 = data.toPandas()
        fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1, subplot_titles=( 'slope', 'intercept', "chi2/dof",  'slope vs intercept' ))
        pau.add_sub_plot(fig,1,1,'1d',[data1['slope']],['slope [mm]','Counts'],[100])
        pau.add_sub_plot(fig,1,2,'1d',[data1['intercept']],['intercept [mm]','Counts'],[100])
        pau.add_sub_plot(fig,2,1,'1d',[data1['a_chi2_reduced']],['chi2/dof','Counts'],xrange=[0,6000,60])
        pau.add_sub_plot(fig,2,2,'2d',[data1['intercept'],data1['slope']],['intercept [mm]','slope [mm]'],[50, 50])
        pau.align_colorbar(fig)
        fig.update_layout( height=700, width=1400, showlegend=False,title_text=f"{input_finename}")
        fig.show()

        if save_flag and savebase is not None:
            saveutil.save_plotly(fig, base_dir=savebase)

    return (check_fit_result,)


@app.cell
def _(fit_gaussian, make_hist, make_subplots, np, pau, saveutil):
    def check_energy_resolution(df_plot1, input_finename="", savebase=None, save_flag=False):

        data = df_plot1.toPandas()
        counts, centers, edges = make_hist(data["Qdiff"], nbins=100, data_range=(-30, 30))
        params, info = fit_gaussian( centers, counts, fit_range=(-20, 20) )
        fitx = np.linspace(-20,20,400)
        fity = _gauss(fitx, params["A"], params["mu"], params["sigma"])

        fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1, subplot_titles=(f"Qdif fit sigma :{params['sigma']}","xg vs slope","xg vs Qdif","slope vs Qdif"))
        pau.add_sub_plot(fig,1,1,'spark-hist',[centers,counts],[r'$\frac{Q_{1}+Q_{3}}{2} - Q_{2} \mathrm{[keV]}$','Counts'],xrange=[100])
        pau.add_sub_plot(fig,1,1,'fit',[fitx, fity])
        pau.add_sub_plot(fig,1,2,'2d',[data["x_ug"], data["slope"]],['wighted x position (upstream) [mm]','slope'],[100,100],[False,False,False],xrange=[-20,20],yrange=[-1,1])
        pau.add_sub_plot(fig,2,1,'2d',[data["x_ug"], data["Qdiff"]],['wighted x position (upstream) [mm]',r'$\frac{Q_{1}+Q_{3}}{2} - Q_{2}$ [keV]'],[100,100],[False,False,False],xrange=[-20,20],yrange=[-20,20])
        pau.add_sub_plot(fig,2,2,'2d',[data["slope"], data["Qdiff"]],['slope',r'$\frac{Q_{1}+Q_{3}}{2} - Q_{2}$ [keV]'],[100,100],[False,False,False],xrange=[-1,1],yrange=[-20,20])
        pau.align_colorbar(fig)
        fig.update_layout( height=700, width=1400, showlegend=False,title_text=f"{input_finename}")
        fig.show()

        if save_flag and savebase is not None:
            saveutil.save_plotly(fig, base_dir=savebase)

    return (check_energy_resolution,)


@app.cell
def _(fit_gaussian, make_hist, make_subplots, np, pau, saveutil):
    def check_tpc_position_charge_correction(df, input_finename="", savebase=None, save_flag=False, a_val=0.22372, b_val=8.56488):

        data = df.toPandas()

        fig = make_subplots(rows=2, cols=3, vertical_spacing=0.15, horizontal_spacing=0.1,
            subplot_titles=( 
                "upstream tpc total charge", 
                "downstream tpc total charge", 
                r"$q_{up} \,\mathrm{vs}\, q_{down}$", 
                "upstream tpc weighted x position", 
                "downstream tpc weighted x position", 
                r"$x_{g,up} \,\mathrm{vs}\, x_{g,down}$", 
            )
        )

        pau.add_sub_plot(fig,1,1,'1d',[data["ΔE_up"]],[r"$Q_\mathrm{sum, up} \mathrm{[keV]}$",r'$\mathrm{Counts}$'],xrange=[10,310,5])
        pau.add_sub_plot(fig,1,2,'1d',[data["ΔE_dn"]],[r"$Q_\mathrm{sum, down} \mathrm{[keV]}$",r'$\mathrm{Counts}$'],xrange=[10,310,5])
        pau.add_sub_plot(fig,1,3,'2d',[data["ΔE_up"],data["ΔE_dn"]],[r"$Q_{sum, up}$",r"$Q_{sum, down}$"],[100, 100], [False,False,False],[10,310],[10,310],True)
        pau.add_sub_plot(fig,2,1,'1d',[data["x_ug"]],[r"$x_{g, up}$",'Counts'],xrange=[-20,20,1])
        pau.add_sub_plot(fig,2,2,'1d',[data["x_dg"]],[r"$x_{g, down}$",'Counts'],xrange=[-20,20,1])
        pau.add_sub_plot(fig,2,3,'2d',[data["x_ug"],data["x_dg"]],[r"$x_{g, up}$",r"$x_{g, down}$"],[100, 100], [False,False,False],[-20,20],[-20,20],True)
        pau.align_colorbar(fig)
        fig.update_layout(height=800, width=1600, showlegend=False,title_text=f"{input_finename}")
        fig.show()

        if save_flag and savebase is not None:
            saveutil.save_plotly(fig, base_dir=savebase)


        df_valid = (df
                    .withColumn("slope", F.try_divide(F.lit(1.0) , F.col("a_slope")))
                    .withColumn("intercept", F.try_divide(F.lit(-1.0) * F.col("a_intercept") , F.col("a_slope")))
                    .filter( F.col("Q_dn") > 10e3 )
                    .filter( F.col("Q_dn") < 30e3 )
                    .filter( F.col("Q_up") > 10e3 )
                    .filter( F.abs(F.col("slope")) < 0.05 )
                    .filter( F.abs(F.col("x_ug")) < 10. )
                )

        data_valid = df_valid.toPandas()


        counts1, centers1, edges1 = make_hist(data_valid["Q_up"], nbins=50, data_range=(10e3, 30e3))
        params1, info1 = fit_gaussian(centers1, counts1, fit_range=(15e3, 25e3) )
        fitx1 = np.linspace(15e3, 25e3,400)
        fity1 = _gauss(fitx1, params1["A"], params1["mu"], params1["sigma"])

        counts2, centers2, edges2 = make_hist(data_valid["Q_dn"], nbins=50, data_range=(20e3, 40e3))
        params2, info2 = fit_gaussian( centers2, counts2, fit_range=(20e3, 30e3) )
        fitx2 = np.linspace(20e3, 30e3,400)
        fity2 = _gauss(fitx2, params2["A"], params2["mu"], params2["sigma"])

        fig = make_subplots(rows=1, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1,
            subplot_titles=( 
                f"upstream tpc total charge (valid), fit result mu: {params1['mu']:.2f} ", 
                f"downstream tpc total charge (valid), fit result mu: {params2['mu']:.2f} "
            )
        )

        pau.add_sub_plot(fig,1,1,'spark-hist',[centers1,counts1],[r'$Q_{\mathrm{up}} {\mathrm{[keV]}$','Counts'])
        pau.add_sub_plot(fig,1,1,'fit',[fitx1, fity1])
        pau.add_sub_plot(fig,1,2,'spark-hist',[centers2,counts2],[r'$Q_{\mathrm{up}} {\mathrm{[keV]}$','Counts'])
        pau.add_sub_plot(fig,1,2,'fit',[fitx2, fity2])

        fig.update_layout(height=600, width=1200, showlegend=False,title_text=f"{input_finename}")
        fig.show()

        deinfo = get_pads()
        de1 = (deinfo.dedxinfo["Am241"]["up"]["de[keV]"] + deinfo.dedxinfo["Cm244"]["up"]["de[keV]"] + deinfo.dedxinfo["Np237"]["up"]["de[keV]"]) / 3.0
        de2 = (deinfo.dedxinfo["Am241"]["down"]["de[keV]"] + deinfo.dedxinfo["Cm244"]["down"]["de[keV]"])/ 2.0

        print(f"[debug] de={de1} a={a_val} b={b_val} mu={params1['mu']}")
        print(f"[debug] de={de2} a={a_val} b={b_val} mu={params2['mu']}")

        g1 = calculate_energy_gain(params1["mu"], dE=de1, a=a_val, b=b_val)
        g2 = calculate_energy_gain(params2["mu"], dE=de2, a=a_val, b=b_val)

        print(f"Gain (P10@20kPa, upstream   tpc): {g1}")
        print(f"Gain (P10@20kPa, downstream tpc): {g2}")

        c1 = calculate_energy_conversion_factor(Gg=g1 ,a=a_val, b=b_val)
        c2 = calculate_energy_conversion_factor(Gg=g2 ,a=a_val, b=b_val)

        print(f"Conversion Factor (P10@20kPa, upstream   tpc): {c1}")
        print(f"Conversion Factor (P10@20kPa, downstream tpc): {c2}")

        if save_flag and savebase is not None:
            saveutil.save_plotly(fig, base_dir=savebase)

    return (check_tpc_position_charge_correction,)


@app.cell
def _(make_subplots, pau, saveutil):
    def check_timing(df_plot6,  input_finename="", savebase=None, save_flag=False):

        data = df_plot6.toPandas()

        fig = make_subplots(rows=2, cols=3, vertical_spacing=0.15, horizontal_spacing=0.1, subplot_titles=(f"y_g,up","y_g,down","test","test","test","test"))
        pau.add_sub_plot(fig,1,1,'1d',[data["y_up_ns"]],["mean y position (upstream) [ns]","Counts"],xrange=[0,1e3,10])
        pau.add_sub_plot(fig,1,2,'1d',[data["y_dn_ns"]],["mean y position (downstream) [ns]","Counts"],xrange=[0,1e3,10])
        pau.add_sub_plot(fig,1,3,'2d',[data["y_up_ns"],data["y_dn_ns"]],['mean y position (upstream) [ns]','mean y position (downstream) [ns]'],[100,100],[False,False,False],xrange=[0,1e3],yrange=[0,1e3])

        pau.add_sub_plot(fig,2,1,'1d',[data["ydif"]],["y dif [ns]","Counts"],xrange=[0,1e3,10])
        pau.add_sub_plot(fig,2,2,'1d',[data["ysum"]],["y sum [ns]","Counts"],xrange=[0,1e3,10])
        pau.add_sub_plot(fig,2,3,'2d',[data["ysum"],data["ydif"]],['y sum [ns]','y dif [ns]'],[100,100],[False,False,False],xrange=[0,1e3],yrange=[0,1e3])

        # pau.add_sub_plot(fig,2,1,'2d',[data["x_ug"], data["Qdiff"]],['wighted x position (upstream) [mm]',r'$\frac{Q_{1}+Q_{3}}{2} - Q_{2}$ [keV]'],[100,100],[False,False,False],xrange=[-20,20],yrange=[-20,20])
        # pau.add_sub_plot(fig,2,2,'2d',[data["slope"], data["Qdiff"]],['slope',r'$\frac{Q_{1}+Q_{3}}{2} - Q_{2}$ [keV]'],[100,100],[False,False,False],xrange=[-1,1],yrange=[-20,20])
        pau.align_colorbar(fig)
        fig.update_layout( height=700, width=1400, showlegend=False,title_text=f"{input_finename}")
        fig.show()

        if save_flag and savebase is not None:
            saveutil.save_plotly(fig, base_dir=savebase)

    return (check_timing,)


@app.cell
def _(fit_gaussian, make_hist, make_subplots, np, pau, saveutil):
    def check_xpos_resolution(df,refz,  input_finename="", savebase=None, save_flag=False):

        data = df.toPandas()
        # .select("x_ug1", "x_ug2", "x_ug3", "x_dg1", "x_dg2", "x_dg3", "a_slope", "a_intercept")
        # .withColumn("slope", F.lit(1.0) / F.col("a_slope"))
        # .withColumn("intercept", F.lit(-1.0) * F.col("a_intercept") / F.col("a_slope"))
                        # .filter( F.abs(F.col("slope")) < 0.05 )
                        # .filter( F.abs(F.col("x_ug")) < 10. )

        offset = -3.031088913245535
        # pad1 = padinfo.get_tpc_info(offset+45)
        # pad2 = padinfo.get_tpc_info(offset+136.5,False)
        miniz1 = offset + 45.
        miniz2 = offset + 136.5
        SSRs = []
        val = 0
        diffz = []


        for i in range(len(data)):

            if ( abs(data["slope"][i]) < 0.05 ) and ( abs(data["x_ug"][i]) < 10) :
                val += ( ( ( data["slope"][i] * miniz1 + data["intercept"][i] ) - data["x_ug"][i] )**2 )
                val += ( ( ( data["slope"][i] * miniz2 + data["intercept"][i] ) - data["x_dg"][i] )**2 )
                # val += np.sqrt( ( ( data["slope"][i] * refz[0] + data["intercept"][i] ) - data["x_ug1"] )**2 )
                # val += np.sqrt( ( ( data["slope"][i] * refz[1] + data["intercept"][i] ) - data["x_ug2"] )**2 )
                # val += np.sqrt( ( ( data["slope"][i] * refz[2] + data["intercept"][i] ) - data["x_ug3"] )**2 )
                # val += np.sqrt( ( ( data["slope"][i] * refz[3] + data["intercept"][i] ) - data["x_dg1"] )**2 )
                # val += np.sqrt( ( ( data["slope"][i] * refz[4] + data["intercept"][i] ) - data["x_dg2"] )**2 )
                # val += np.sqrt( ( ( data["slope"][i] * refz[5] + data["intercept"][i] ) - data["x_dg3"] )**2 )
                SSRs.append(np.sqrt(val))
                diffz.append(abs(data["x_ug"][i]-data["x_dg"][i]))

                val = 0




        fig = make_subplots(rows=2, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1, subplot_titles=(f"Sum of RMS",))
        counts, centers, edges = make_hist(SSRs, nbins=200, data_range=(-1, 4))
        params, info = fit_gaussian( centers, counts, fit_range=(-20, 20) )
        fitx = np.linspace(-20,20,400)  
        fity = _gauss(fitx, params["A"], params["mu"], params["sigma"])
        print(params["sigma"])

        pau.add_sub_plot(fig,1,1,'spark-hist',[centers,counts],['SSR','Counts'],xrange=[100])
        # pau.add_sub_plot(fig,1,1,'fit',[fitx, fity])

        counts, centers, edges = make_hist(diffz, nbins=300, data_range=(-50, 50))
        params, info = fit_gaussian( centers, counts, fit_range=(-20, 20) )
        fitx = np.linspace(-20,20,400)  
        fity = _gauss(fitx, params["A"], params["mu"], params["sigma"])
        print(params["sigma"])
        pau.add_sub_plot(fig,1,2,'spark-hist',[centers,counts],['SSR','Counts'],xrange=[100])
        # pau.add_sub_plot(fig,1,2,'fit',[fitx, fity])
        # pau.add_sub_plot(fig,1,1,'fit',[fitx, fity])
        # pau.add_sub_plot(fig,1,2,'2d',[data["x_ug"], data["slope"]],['wighted x position (upstream) [mm]','slope'],[100,100],[False,False,False],xrange=[-20,20],yrange=[-1,1])
        # pau.add_sub_plot(fig,2,1,'2d',[data["x_ug"], data["Qdiff"]],['wighted x position (upstream) [mm]',r'$\frac{Q_{1}+Q_{3}}{2} - Q_{2}$ [keV]'],[100,100],[False,False,False],xrange=[-20,20],yrange=[-20,20])
        # pau.add_sub_plot(fig,2,2,'2d',[data["slope"], data["Qdiff"]],['slope',r'$\frac{Q_{1}+Q_{3}}{2} - Q_{2}$ [keV]'],[100,100],[False,False,False],xrange=[-1,1],yrange=[-20,20])
        # pau.align_colorbar(fig)
        fig.update_layout( height=700, width=1400, showlegend=False,title_text=f"{input_finename}")
        fig.show()

        if save_flag and savebase is not None:
            saveutil.save_plotly(fig, base_dir=savebase)

    return (check_xpos_resolution,)


@app.function
def csum(cond, expr):
    return F.sum(F.when(cond, expr).otherwise(F.lit(0.0)))


@app.function
def ccount(cond):
    return F.sum(F.when(cond, F.lit(1)).otherwise(F.lit(0)))


@app.function
def cmean(cond, expr):
    num = F.sum(F.when(cond, F.col(expr) if isinstance(expr, str) else expr).cast("double"))
    den = F.sum(F.when(cond, F.lit(1)).otherwise(F.lit(0))).cast("double")
    # 分母が 0 のときは NULL（安全）
    return F.when(den > 0, num / den).otherwise(F.lit(None).cast("double"))


@app.cell
def _(
    check_energy_resolution,
    check_fit_result,
    check_hq,
    check_nq,
    check_timing,
    check_tpc_map,
    check_tpc_position_charge_correction,
    check_xpos_resolution,
    np,
    odr_group,
    odr_schema,
    os,
    plt,
    saveutil,
    this_file_path,
    toml,
):
    def main(maxevt):

        toml_file_path = this_file_path  / "../../../parameters.toml"

        with open(toml_file_path, "r") as f:
            config = toml.load(f)

        analysinfo = config["analysis"]
        fileinfo = config["fileinfo"]
        map_path = analysinfo["tpc_mapfile"]
        base_path = fileinfo["base_output_path"]  + "/" + fileinfo["input_file_name"] 
        input = base_path + "_phys.parquet"
        input_finename = os.path.basename(input)
        savebase = str(this_file_path.parent.parent.parent / "figs")

        schema_map = (
            T.StructType()
            .add("sampaNo",   T.IntegerType())
            .add("sampaID",   T.IntegerType())
            .add("samidareID",T.IntegerType())
            .add("tpcID",     T.IntegerType())
            .add("padID",     T.IntegerType())
            .add("gid",       T.StringType())  
        )

        spark = (
            SparkSession.builder
            .config("spark.driver.memory", "8g")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "128") 
            .config("spark.sql.parquet.enableVectorizedReader", "false")
            .config("spark.sql.files.maxPartitionBytes", 32 * 1024 * 1024)
            .getOrCreate()
        )

        df_map_raw = (spark.read
            .option("header", True)
            .schema(schema_map)
            .csv(map_path)
        )

        df_map = (df_map_raw
            .withColumn(
                "gid_mapped",
                F.when(F.col("gid") == F.lit("G"), F.lit(-1)).otherwise(F.col("gid").cast("int"))
            )
            .select(
                F.col("samidareID").cast("long").alias("map_samidare_id"),
                F.col("gid_mapped")
            )
        )

        #################### load data ####################
        df = spark.read.parquet(input)    

        #################### convert to physics values ####################
        coef_up_a,     coef_up_b = 0.007594825219782722, 0.2907597292526938
        coef_down_a, coef_down_b = 0.006512604389126628, 0.24932806669203858

        if maxevt >= 0:
            df_gid = (df
                .join(F.broadcast(df_map), df["samidare_id"] == df_map["map_samidare_id"], "left")
                .drop("map_samidare_id")
                .withColumnRenamed("gid_mapped", "tpc_id")
                .withColumn("maxsample_index_raw", F.array_position(F.col("pulse"), F.array_max(F.col("pulse"))))
                .withColumn("maxsample_index", F.when(F.col("maxsample_index_raw") <= 0, F.lit(None).cast("int")).otherwise(F.col("maxsample_index_raw").cast("int")))
                .withColumn("maxsample_timing_ms", (F.element_at(F.col("times"), F.col("maxsample_index")) / F.lit(320000.0)).cast("double"))
                .withColumn(
                    "energy_deposit_keV",
                    F.when(F.col("tpc_id").between( 0,  59), F.col("charge") * F.lit(coef_up_a) + F.lit(coef_up_b)) 
                    .when(F.col("tpc_id" ).between(60, 119), F.col("charge") * F.lit(coef_down_a) + F.lit(coef_down_b))
                    .otherwise(F.lit(None).cast("double"))
                )
                .limit(maxevt)
            )
        else:
            df_gid = (df
                .join(F.broadcast(df_map), df["samidare_id"] == df_map["map_samidare_id"], "left")
                .drop("map_samidare_id")
                .withColumnRenamed("gid_mapped", "tpc_id")
                .withColumn("maxsample_index_raw", F.array_position(F.col("pulse"), F.array_max(F.col("pulse"))))
                .withColumn("maxsample_index", F.when(F.col("maxsample_index_raw") <= 0, F.lit(None).cast("int")).otherwise(F.col("maxsample_index_raw").cast("int")))
                .withColumn("maxsample_timing_ms", (F.element_at(F.col("times"), F.col("maxsample_index")) / F.lit(320000.0)).cast("double"))
                .withColumn(
                    "energy_deposit_keV",
                    F.when(F.col("tpc_id").between( 0,  59), F.col("charge") * F.lit(coef_up_a) + F.lit(coef_up_b)) 
                    .when(F.col("tpc_id" ).between(60, 119), F.col("charge") * F.lit(coef_down_a) + F.lit(coef_down_b))
                    .otherwise(F.lit(None).cast("double"))
                )
            )

        if 0:
            dft =  df_gid.select("energy_deposit_keV")
            data = dft.toPandas()
            plt.hist(data.energy_deposit_keV, bins=100)
            plt.show()  

        tpcs = get_pads()
        pad_rows = [(int(tid), float(c[0]), float(c[2])) for tid, c in zip(tpcs.ids, tpcs.centers)]
        df_pad = spark.createDataFrame(pad_rows, schema="tpc_id int, x double, z double")

        #################### event build ####################
        df_hit = (df_gid
            .join(F.broadcast(df_pad), on="tpc_id", how="left")
            .withColumn("w", F.col("charge").cast("double"))
            .withColumn("is_up", (F.col("tpc_id").between(0, 59)).cast("int"))
            .withColumn("is_dn", (F.col("tpc_id") >= 60).cast("int"))
            .withColumn("is_al", (F.col("tpc_id").between(0, 119)).cast("int"))
            .withColumn("is_up1", (F.col("tpc_id").between( 0 ,  19)).cast("int"))
            .withColumn("is_up2", (F.col("tpc_id").between(20 ,  39)).cast("int"))
            .withColumn("is_up3", (F.col("tpc_id").between(40 ,  59)).cast("int")) 
            .withColumn("is_dn1", (F.col("tpc_id").between(60 ,  79)).cast("int"))
            .withColumn("is_dn2", (F.col("tpc_id").between(80 ,  99)).cast("int"))
            .withColumn("is_dn3", (F.col("tpc_id").between(100, 119)).cast("int")) 
            .filter(F.col("is_al") == 1)   
        )

        df_evt = (df_hit
            .groupBy("event_id")
            .agg(
                F.sort_array(F.collect_list(F.struct(
                    F.col("tpc_id").alias("k"),            
                    F.col("x").cast("double").alias("x"),
                    F.col("maxsample_timing_ms").cast("double").alias("y"),
                    F.col("z").cast("double").alias("z"),
                    F.col("energy_deposit_keV").alias("e")
                ))).alias("hits"),

                cmean(F.col("is_up")==1, F.col("z")).alias("z_up"),
                cmean(F.col("is_dn")==1, F.col("z")).alias("z_dn"),

                csum(F.col("is_al")==1, F.col("w")*F.col("x")).alias("Wx"),
                csum(F.col("is_up")==1, F.col("w")*F.col("x")).alias("Wx_up"),
                csum(F.col("is_dn")==1, F.col("w")*F.col("x")).alias("Wx_dn"),

                csum(F.col("is_up1")==1, F.col("w")*F.col("x")).alias("Wx_up1"),
                csum(F.col("is_up2")==1, F.col("w")*F.col("x")).alias("Wx_up2"),
                csum(F.col("is_up3")==1, F.col("w")*F.col("x")).alias("Wx_up3"),

                csum(F.col("is_dn1")==1, F.col("w")*F.col("x")).alias("Wx_dn1"),
                csum(F.col("is_dn2")==1, F.col("w")*F.col("x")).alias("Wx_dn2"),
                csum(F.col("is_dn3")==1, F.col("w")*F.col("x")).alias("Wx_dn3"),

                ccount(F.col("is_al")==1).alias("N_al"),
                ccount(F.col("is_up")==1).alias("N_up"),
                ccount(F.col("is_dn")==1).alias("N_dn"),

                csum(F.col("is_al")==1, F.col("w")).alias("Q_al"),
                csum(F.col("is_up")==1, F.col("w")).alias("Q_up"),
                csum(F.col("is_dn")==1, F.col("w")).alias("Q_dn"),

                csum(F.col("is_up1")==1, F.col("w")).alias("Q_up1"),
                csum(F.col("is_up2")==1, F.col("w")).alias("Q_up2"),
                csum(F.col("is_up3")==1, F.col("w")).alias("Q_up3"),
                csum(F.col("is_dn1")==1, F.col("w")).alias("Q_dn1"),
                csum(F.col("is_dn2")==1, F.col("w")).alias("Q_dn2"),
                csum(F.col("is_dn3")==1, F.col("w")).alias("Q_dn3"),

                csum(F.col("is_al")==1, F.col("energy_deposit_keV")).alias("ΔE_al"),
                csum(F.col("is_up")==1, F.col("energy_deposit_keV")).alias("ΔE_up"),
                csum(F.col("is_dn")==1, F.col("energy_deposit_keV")).alias("ΔE_dn"),

                csum(F.col("is_up1")==1, F.col("energy_deposit_keV")).alias("ΔE_up1"),
                csum(F.col("is_up2")==1, F.col("energy_deposit_keV")).alias("ΔE_up2"),
                csum(F.col("is_up3")==1, F.col("energy_deposit_keV")).alias("ΔE_up3"),
            )

            .withColumn("tpc_ids",  F.expr("transform(hits, h -> int(h.k))"))
            .withColumn("xs",       F.expr("transform(hits, h -> double(h.x))"))
            .withColumn("ys",       F.expr("transform(hits, h -> double(h.y))"))
            .withColumn("zs",       F.expr("transform(hits, h -> double(h.z))"))
            .withColumn("energy_deposits_keV", F.expr("transform(hits, h -> double(h.e))"))

            .drop("hits")

            .withColumn("x_ug", F.col("Wx_up") / F.nullif(F.col("Q_up"), F.lit(0.0)))
            .withColumn("x_dg", F.col("Wx_dn") / F.nullif(F.col("Q_dn"), F.lit(0.0)))

            .withColumn("x_ug1", F.col("Wx_up1") / F.nullif(F.col("Q_up1"), F.lit(0.0)))
            .withColumn("x_ug2", F.col("Wx_up2") / F.nullif(F.col("Q_up2"), F.lit(0.0)))
            .withColumn("x_ug3", F.col("Wx_up3") / F.nullif(F.col("Q_up3"), F.lit(0.0)))
            .withColumn("x_dg1", F.col("Wx_dn1") / F.nullif(F.col("Q_dn1"), F.lit(0.0)))
            .withColumn("x_dg2", F.col("Wx_dn2") / F.nullif(F.col("Q_dn2"), F.lit(0.0)))
            .withColumn("x_dg3", F.col("Wx_dn3") / F.nullif(F.col("Q_dn3"), F.lit(0.0)))

            .withColumn("y_offset", F.coalesce( F.expr("array_min(filter(ys, x -> x IS NOT NULL))"), F.lit(0.0) ))

            .withColumn("ys0", F.expr("transform(ys, y -> CASE WHEN y IS NULL THEN NULL ELSE y - y_offset END)") )

            .withColumn("zup", F.expr("filter(arrays_zip(tpc_ids, ys0), z -> z.tpc_ids BETWEEN 0 AND 59 AND z.ys0 IS NOT NULL)") )
            .withColumn("zdn", F.expr("filter(arrays_zip(tpc_ids, ys0), z -> z.tpc_ids >= 60 AND z.ys0 IS NOT NULL)") )

            .withColumn("y_up_sum", F.expr("aggregate(zup, 0.0D, (acc, z) -> acc + z.ys0)"))
            .withColumn("y_dn_sum", F.expr("aggregate(zdn, 0.0D, (acc, z) -> acc + z.ys0)"))

            .withColumn("y_up_n", F.size("zup"))
            .withColumn("y_dn_n", F.size("zdn"))

            .withColumn("y_up_ns", F.when( F.col("y_up_n") > 0, (F.col("y_up_sum") / F.col("y_up_n")) * F.lit(1e6) ).otherwise(F.lit(None).cast("double")))
            .withColumn("y_dn_ns", F.when( F.col("y_dn_n") > 0, (F.col("y_dn_sum") / F.col("y_dn_n")) * F.lit(1e6) ).otherwise(F.lit(None).cast("double")))

            .drop("zup", "zdn", "y_up_sum", "y_dn_sum", "y_up_n", "y_dn_n")
        )


        #################### tracking ####################
        df_for_fit = (df_hit
            .select("event_id", "x", "z", "tpc_id", F.col("w").cast("double").alias("w"))
            .filter(F.col("w") > 0)
        )

        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        df_odr = df_for_fit.repartition("event_id").groupBy("event_id").applyInPandas(odr_group, schema=odr_schema)
        df_evt = df_evt.join(df_odr, on=["event_id"], how="left")

        #################### plot ####################

        if 0:
            df_plot = (df_gid.filter(df_gid['tpc_id'] >= 0).orderBy("maxsample_timing_ms"))
            check_tpc_map(df_plot, input_finename)

        if 0:
            df_plot1 = (df_gid.filter(df_gid['tpc_id'] >= 0).filter( F.col("charge") < 4000)).select("charge","maxsample")
            df_plot2 = (df_gid.filter(df_gid['tpc_id'] >= 0)).select("charge","maxsample")
            check_hq(df_plot1, df_plot2) 

        if 0:
            check_nq(df_evt)

        if 0:
            df_plot7 = (df_evt
                .filter( F.col("Q_up") < 30e3 )
                .filter( F.col("Q_dn") > 20e3 )
                .filter( F.col("N_dn") < 13 )
                .filter( F.col("N_up") < 13 )
                )

            if 0:
                basesavepath = f"{savebase}/20250914"
                check_hit_patturn(df_plot7, savefilepath=basesavepath, dpi=600, maxevt=120)
            else:
                check_hit_patturn(df_plot7)

        if 0:
            df_plot3 = (df_evt
                        .select("a_slope","a_intercept","a_dof","a_chi2_reduced")
                        .filter( F.col("Q_up") < 30e3 )
                        .filter( F.col("Q_dn") > 20e3 )
                        .withColumn("slope", F.lit(1.0) / F.col("a_slope"))
                        .withColumn("intercept", F.lit(-1.0) * F.col("a_intercept") / F.col("a_slope"))
                        )
            check_fit_result(df_plot3)

        if 0:
            df_plot4 = (df_evt.select("x_ug","x_dg","ΔE_up","ΔE_dn","a_slope","a_intercept","Q_up", "Q_dn"))
            check_tpc_position_charge_correction(df_plot4, a_val=0.22372, b_val=8.56488)

            df_plot5 = (df_evt
                        .select("ΔE_up1","ΔE_up2","ΔE_up3","x_ug","a_slope","a_intercept")
                        .withColumn("Qdiff", ( F.col("ΔE_up1") + F.col("ΔE_up3") ) / F.lit(2.) - F.col("ΔE_up2"))
                        .withColumn("Qtot", ( F.col("ΔE_up1") + F.col("ΔE_up3") + F.col("ΔE_up2") ))
                        .withColumn("slope", F.try_divide(F.lit(1.0) , F.col("a_slope")))
                        .filter( F.abs(F.col("slope")) < 0.05 )
                        .filter( F.abs(F.col("x_ug")) < 10. )
                    )
            check_energy_resolution(df_plot5)

        if 0:
            df_plot6 = (df_evt
                        .select("y_up_ns", "y_dn_ns", "x_ug", "a_slope","N_up","N_dn")
                        .withColumn("slope", F.try_divide(F.lit(1.0) , F.col("a_slope")))
                        .withColumn("ydif", F.col('y_up_ns') - F.col("y_dn_ns"))
                        .withColumn("ysum", F.col('y_up_ns') + F.col("y_dn_ns"))
                        .filter( F.abs(F.col("slope")) < 1 )
                        .filter( F.abs(F.col("x_ug"))  < 12.  )
                        .filter( F.abs(F.col("N_up"))  > 3    )
                        .filter( F.abs(F.col("N_dn"))  > 3    )
                    )

            check_timing(df_plot6)

        if 1:
            refzs = []
            sumz1,sumz2,sumz3,sumz4,sumz5,sumz6 = [],[],[],[],[],[]

            for i in range(len(tpcs.ids)):
                # print(tpcs.ids[i], tpcs.centers[i])

                if tpcs.ids[i] <= 19:
                    sumz1.append(tpcs.centers[i][2])

                elif tpcs.ids[i] <= 39:
                    sumz2.append(tpcs.centers[i][2])

                elif tpcs.ids[i] <= 59:
                    sumz3.append(tpcs.centers[i][2])

                elif tpcs.ids[i] <= 79:
                    sumz4.append(tpcs.centers[i][2])

                elif tpcs.ids[i] <= 99:
                    sumz5.append(tpcs.centers[i][2])

                elif tpcs.ids[i] <= 119:
                    sumz6.append(tpcs.centers[i][2])

            refzs.append(np.mean(sumz1))
            refzs.append(np.mean(sumz2))
            refzs.append(np.mean(sumz3))
            refzs.append(np.mean(sumz4))
            refzs.append(np.mean(sumz5))
            refzs.append(np.mean(sumz6))

            for i in range(len(refzs)):
                print(i,refzs[i])

            df_plot7 = (df_evt
                        .filter( F.col("Q_up") < 30e3 )
                        .filter( F.col("Q_dn") > 20e3 )
                        .filter( F.col("N_dn") < 13 )
                        .filter( F.col("N_up") < 13 )
                        .select("x_ug1", "x_ug2", "x_ug3", "x_dg1", "x_dg2", "x_dg3", "a_slope", "a_intercept", "x_ug","x_dg")
                        .withColumn("slope", F.lit(1.0) / F.col("a_slope"))
                        .withColumn("intercept", F.lit(-1.0) * F.col("a_intercept") / F.col("a_slope"))
                        )

            check_xpos_resolution(df_plot7, refzs)


        if 0:
            print(f"Schema (df) input data")
            df.printSchema()

            print(f"Schema (df_map) tpc map file")
            df_map.printSchema()

            print(f"Schema (df_gid) convert several data and add tpc id")
            df_gid.printSchema()

            print(f"Schema (df_hit) gropu event id")
            df_hit.printSchema()

            print(f"Schema (df_odr) tracking")
            df_odr.printSchema()

            print(f"Schema (df_evt) merge df_odr and df_hit")
            df_evt.printSchema()

        if 0:
            saveutil.generate_gif(
                input_dir=f"{savebase}/20250914",   
                output_dir=f"{savebase}/gifs",      
                duration_s=0.2,                   
                pattern="*.png",          
                sort_by="ctime",               
            )

    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # code開発
    """)
    return


@app.cell
def _():
    spark = (
            SparkSession.builder
            .config("spark.driver.memory", "8g")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "128") 
            .config("spark.sql.parquet.enableVectorizedReader", "false")
            .config("spark.sql.files.maxPartitionBytes", 32 * 1024 * 1024)
            .getOrCreate()
        )

    spark.sparkContext.setLogLevel("ERROR")
    return (spark,)


@app.cell
def _():
    map_path = 'prm/cat/minitpc.map'
    input = 'output/minitpc_demo_pulse.parquet'
    return input, map_path


@app.cell
def _():
    schema_map = (
        T.StructType()
        .add("sampaNo",   T.IntegerType())
        .add("sampaID",   T.IntegerType())
        .add("samidareID",T.IntegerType())
        .add("tpcID",     T.IntegerType())
        .add("padID",     T.IntegerType())
        .add("gid",       T.StringType())  
    )
    return (schema_map,)


@app.cell
def _(map_path, schema_map, spark):
    df_map_raw = (spark.read
        .option("header", True)
        .schema(schema_map)
        .csv(map_path)
    )

    df_map = (df_map_raw
        .withColumn(
            "gid_mapped",
            F.when(F.col("gid") == F.lit("G"), F.lit(-1)).otherwise(F.col("gid").cast("int"))
        )
        .select(
            F.col("samidareID").cast("long").alias("map_samidare_id"),
            F.col("gid_mapped"),
            F.col("tpcID").cast("int").alias("dev_id"),
            F.col("padID").cast("int").alias("pad_id")
        )

    )


    df_map.show(10)
    return (df_map,)


@app.cell
def _(input, spark):
    df = spark.read.parquet(input)  

    df = df.withColumn("samidare_id", F.col("chip") * F.lit(32) + F.col("channel") )

    df.show(10)
    return (df,)


@app.cell
def _(df, df_map):
    df_gid = ( df
        ####################################################
        .join(
            F.broadcast(df_map),
            df["samidare_id"] == df_map["map_samidare_id"], 
            "left"
        )
        ####################################################
        .drop("map_samidare_id")
        ####################################################
        .withColumnRenamed("gid_mapped", "tpc_id")
        ####################################################
        # .withColumn(
        #     "energy_deposit_keV",
        #     F.when(F.col("tpc_id").between( 0,  59), F.col("charge") * F.lit(coef_up_a) + F.lit(coef_up_b)) 
        #     .when(F.col("tpc_id" ).between(60, 119), F.col("charge") * F.lit(coef_down_a) + F.lit(coef_down_b))
        #     .otherwise(F.lit(None).cast("double"))
        # )
    )
    return (df_gid,)


@app.cell
def _(df_gid):
    a1, b1 = 0.007594825219782722, 0.2907597292526938
    a2, b2 = 0.006512604389126628, 0.24932806669203858

    df_tpc1 = df_gid.filter(F.col("dev_id")==1)
    df_tpc2 = df_gid.filter(F.col("dev_id")==2)

    df_evt1 = dev.event_builder.build_events(
        df_tpc1,
        time_col='timestamp_ns',
        id_col='event_id',
        threshold=1.5e6,
        device_name='mini1'
    )



    df_evt2 = dev.event_builder.build_events(
        df_tpc2,
        time_col='timestamp_ns',
        id_col='event_id',
        threshold=1.5e6,
        device_name='mini2'
    )

    df_evt1 = df_evt1.withColumn("de_mini1", F.col("qtot_mini1") * F.lit(a1) + F.lit(b1))
    df_evt2 = df_evt2.withColumn("de_mini2", F.col("qtot_mini2") * F.lit(a2) + F.lit(b2))

    # df_evt = df_evt1.join(df_evt2.drop("timestamp_ns"), on=['event_id'])
    return df_evt1, df_evt2


@app.cell
def _(df_evt1, df_evt2):
    from plotly.subplots import make_subplots

    # _nmax = 20000

    _df = ( df_evt1
        .orderBy("timestamp_ns")
        .toPandas()

    )
    _c1 = _df["de_mini1"].to_numpy()

    _df = ( df_evt2
        .orderBy("timestamp_ns")
        .toPandas()

    )
    _c2 = _df["de_mini2"].to_numpy()


    _fig = make_subplots(1,2,subplot_titles=(["A","B"]))

    molib.plot.add_sub_plot(
        _fig,1,1,
        data=[_c1 ],
        func=molib.plot.go_Histogram,
        xrange=[0,300,3]

    )

    molib.plot.add_sub_plot(
        _fig,1,2,
        data=[_c2],
        func=molib.plot.go_Histogram,
        xrange=[0,300,3]
    )

    _fig.update_layout(height=300, width=1000, showlegend=True)
    return (make_subplots,)


@app.function
def reevent_by_time_window(
    parquet1,
    parquet2,
    *,
    window_ns,                 # 例: 1000 (1usなら 1000ns, 10usなら 10_000ns など)
    ts_col: str = "timestamp_ns",
    out_event_col: str = "merged_event_id",
):
    # 読み込み + 出どころを付与
    a = parquet1.withColumn("dev", F.lit("mini1"))
    b = parquet2.withColumn("dev", F.lit("mini2"))

    df = a.unionByName(b, allowMissingColumns=True)

    # timestamp を整数キーに（double の完全一致問題回避）
    df = df.withColumn("merged_ts", F.col(ts_col).cast("long"))

    # 同一時刻の安定タイブレーク（順序を安定化）
    df = df.withColumn("__rid", F.monotonically_increasing_id())

    w = Window.orderBy(F.col("merged_ts").asc(), F.col("__rid").asc())

    prev_ts = F.lag("merged_ts").over(w)
    dt = F.col("merged_ts") - prev_ts

    # dt が null（先頭）または window_ns より大きければ新イベント
    is_new = F.when(prev_ts.isNull() | (dt > F.lit(window_ns)), 1).otherwise(0)

    df2 = (
        df
        .withColumn(out_event_col, (F.sum(is_new).over(w) - 1).cast("long"))
        .drop("__rid")  # 必要なら残してもOK
    )
    return df2


@app.cell
def _(df_evt1, df_evt2):
    df_re = reevent_by_time_window(
        parquet1=df_evt1,
        parquet2=df_evt2,
        window_ns=3e6,
    )

    df_re.orderBy("merged_ts").show(50, truncate=True)
    return (df_re,)


@app.function
def event_group(df, dev="mini1", debug=False):
    _dev = dev

    _cols_to_lift = [
        "event_id",
        "timestamp_ns",
        f"chip_{_dev}",
        f"channel_{_dev}",
        f"charge_{_dev}",
        f"peak_{_dev}",
        f"baseline_{_dev}",
        f"pulse_{_dev}",
        f"time_{_dev}",
        f"qtot_{_dev}",
        f"de_{_dev}",
        "merged_ts",
        "merged_event_id",
    ]

    df_grouped = (
        df
        .orderBy("merged_ts")
        .filter(F.col("dev") == _dev)
        .select(*_cols_to_lift )
        .groupBy("merged_event_id")
        .agg(
            F.sort_array(
                F.collect_list(F.struct(*_cols_to_lift))
            ).alias("rows")
        )
        .select(
            "merged_event_id",
            *[F.expr(f"transform(rows, x -> x.{c})").alias(c) for c in _cols_to_lift[:-1]],
        )
        .withColumnRenamed("event_id", f"event_id_{_dev}")
        .withColumnRenamed("timestamp_ns", f"timestamp_ns_{_dev}")
        .withColumnRenamed("merged_ts", f"merged_ts_{_dev}")
    )

    if debug:
        df_grouped.show(10)
        df_grouped.printSchema()

    return df_grouped


@app.cell
def _(df_re):
    df_evt1_m = event_group(df_re, 'mini1')
    df_evt2_m = event_group(df_re, 'mini2')

    df_joined_evt = df_evt1_m.join(df_evt2_m, on="merged_event_id", how="left")
    df_joined_evt.filter(F.col("event_id_mini2").isNotNull()).show(10)
    return (df_joined_evt,)


@app.cell
def _(df_joined_evt, make_subplots):
    _df = ( df_joined_evt
        .filter(F.col("event_id_mini2").isNotNull())
        .filter(F.size("de_mini1") == 1)
        .filter(F.size("de_mini2") == 1)
        .select(
            "de_mini1",
            "de_mini2", 
            "charge_mini1", 
            "charge_mini2", 
            "chip_mini1", 
            "chip_mini2", 
            "channel_mini1", 
            "channel_mini2",
            "charge_mini1",
            "charge_mini2"
        )
        .withColumn("de1", F.explode("de_mini1"))
        .withColumn("de2", F.explode("de_mini2"))
        .withColumn("_1", F.explode("charge_mini1"))
        .withColumn("_2", F.explode("charge_mini2"))
        .withColumn("n1", F.size("_1"))
        .withColumn("n2", F.size("_2"))   
        .withColumn("chip1", F.explode("chip_mini1"))
        .withColumn("chip2", F.explode("chip_mini2"))
        .withColumn("channel1", F.explode("channel_mini1"))
        .withColumn("channel2", F.explode("channel_mini2"))
        .withColumn("q1", F.explode("charge_mini1"))
        .withColumn("q2", F.explode("charge_mini2"))
    )

    df_hit = ( _df
        .select("de1","de2","n1", "n2", "chip1", "chip2", "channel1", "channel2", "q1", "q2")
        # .filter(F.col("n1") > 8)
        .filter(F.col("n2") > 8)
    )

    _df =  df_hit.toPandas()

    _c1 = _df ["de1"].to_numpy()
    _c2 = _df ["de2"].to_numpy()
    _n1 = _df ["n1"].to_numpy()
    _n2 = _df ["n2"].to_numpy()

    _fig = make_subplots(1,2,subplot_titles=(["A","B",'c','d']))

    molib.plot.add_sub_plot(
        _fig,1,1,
        data=[_c1 ],
        func=molib.plot.go_Histogram,
        xrange=[0,300, 3]

    )

    molib.plot.add_sub_plot(
        _fig,1,2,
        data=[_c2],
        func=molib.plot.go_Histogram,
        xrange=[0,300, 3]
    )


    # molib.plot.add_sub_plot(
    #     _fig,2,1,
    #     data=[_n1, _c1 ],
    #     func=molib.plot.go_Heatmap,
    #     bins=[15,100],
    #     xrange=[0,15],
    #     yramge=[0,300]
    # )

    # molib.plot.add_sub_plot(
    #     _fig,2,2,
    #     data=[_n2, _c2 ],
    #     func=molib.plot.go_Heatmap,
    #     bins=[15,100],
    #     xrange=[0,15],
    #     yramge=[0,300]
    # )


    _fig.update_layout(height=300, width=1000, showlegend=True)
    return (df_hit,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    def get_color_bins(values, cmap_name="viridis", fmt="hex", n_bins=10):
        values = np.asarray(values)
        if values.size == 0:
            return np.array([]), []

        if n_bins <= 0:
            raise ValueError("n_bins must be >= 1")

        vmin = float(np.min(values))
        vmax = float(np.max(values))

        if np.isclose(vmin, vmax):
            # 1bin
            edges = np.array([vmin, vmax])
            n_bins_eff = 1
        else:
            edges = np.linspace(vmin, vmax, n_bins + 1)
            n_bins_eff = n_bins

        cmap = plt.get_cmap(cmap_name, n_bins_eff)
        rgba = [cmap(i) for i in range(n_bins_eff)]

        if fmt == "hex":
            colors = [to_hex(c[:3]) for c in rgba]
        elif fmt == "rgb":
            colors = [tuple(int(round(ch * 255)) for ch in c[:3]) for c in rgba]
        else:
            raise ValueError("fmt must be 'hex' or 'rgb'")

        return edges, colors

    return get_color_bins, np, plt


@app.cell
def _(np):
    def get_color_array(values, edges, colors):
        """
        edges: 長さ n_bins+1 の境界配列
        colors: 長さ n_bins の色リスト
        """
        values = np.asarray(values, dtype=float)
        edges = np.asarray(edges, dtype=float)

        if values.size == 0:
            return []

        # bin index を計算: 0..n_bins-1
        idx = np.digitize(values, edges, right=False) - 1
        idx = np.clip(idx, 0, len(colors) - 1)

        return [colors[i] for i in idx]

    return (get_color_array,)


@app.cell
def _(df_hit, get_color_array, get_color_bins):
    ####
    tpc_map = 'prm/cat/minitpc.map'

    mapdf = pd.read_csv(tpc_map)
    mapdf['padID'] = pd.to_numeric(mapdf['padID'], errors='coerce')
    mapdf['gid'] = pd.to_numeric(mapdf['gid'], errors='coerce')

    mapdf['tpcID'] = mapdf['tpcID'].astype(int)
    mapdf['padID'] = mapdf['padID'].fillna(-1).astype(int)
    mapdf['gid'] = mapdf['gid'].fillna(-1).astype(int)
    mapdf['sampaNo'] = mapdf['sampaNo'].astype(int)
    mapdf['sampaID'] = mapdf['sampaID'].astype(int)
    mapdf['samidareID'] = mapdf['samidareID'].astype(int)

    mapdf = mapdf.reset_index(drop=True)
    # mapdf.head(10)

    ####
    offset = -3.031088913245535
    pad1 = util.padinfo.get_tpc_info(offset+45)
    pad2 = util.padinfo.get_tpc_info(offset+136.5,False)
    tpcs = util.padinfo.marge_padinfos(pad1,pad2)

    # tpcs.ids

    ####
    _df = ( df_hit
        .withColumn(
            "samidare_id1",
            F.expr("transform(arrays_zip(chip1, channel1), x -> x.chip1 * 32 + x.channel1)")
        )
        .withColumn(
            "samidare_id2",
            F.expr("transform(arrays_zip(chip2, channel2), x -> x.chip2 * 32 + x.channel2)")
        )
    )

    hits_data = []


    for _ev in _df.toLocalIterator():
        _sid = _ev['samidare_id1'] + _ev['samidare_id2']
        _q = _ev['q1'] + _ev['q2']

        _gid = []

        for _id in _sid:
            _gid.append(util.mapfile.get_any_from_mapdf_using_ref(mapdf, refLabel='samidareID', refID=_id, label='gid'))

        # tpcid_arr = _gid
        # de_arr    = _q 

        # reflist = de_arr
        _q_lst = [0] * len(tpcs.ids)

        for _i in range(len(_gid)):
            _q_lst[_gid[_i]] = _q[_i]


        _edges, _colors = get_color_bins(_q_lst, n_bins=20, cmap_name="ocean_r", fmt="hex")
        _color_array = get_color_array(_q_lst, _edges, _colors)


        hits_data.append(_color_array)    
    return hits_data, tpcs


@app.cell
def _(hits_data, mo):
    # 描画したいイベントの index（例：0..999）
    idx = mo.ui.slider(start=0, stop=len(hits_data)-1, step=1, value=0, label="event index")
    idx
    return (idx,)


@app.cell
def _(hits_data, idx, tpcs):
    tpcs.show_pads(
        plot_type='map',
        plane='zx',
        color_map=hits_data[idx.value], 
        check_id = False,
        check_size=3,
        check_data = tpcs.ids,
        canvassize = [8,7],
        return_flag=True
    )
    return


@app.cell
def _(spark):
    spark.stop()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
