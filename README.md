# My-Complex-Steerable-Pyramid
## そもそも、Complex Steerable Pyramid とは？
Complex Steerable Pyramid は Steerable Pyramid を効率化した画像ピラミッドです。  
その大元である Steerable Pyramid は多重解像度解析(画像ピラミッドと読み替えても良いです)に方向の概念を導入したものです。  
画像ピラミッドの各レベル(すなわち各解像度)のDFTに極座標系を利用して作成した角度マスクを掛けることで、画像ピラミッドを多方向に分解します。
Complex Steerable Pyramid ではさらに局所的な位相の情報を含めることができます。これによって位相を利用したテクスチャ生成や動きの増幅などに使えるそうです。

## このソースコードについて
My-Complex-Steerable-Pyramid として公開するソースコードは、Isaac Berrios氏が公開している資料をもとに作成しました。彼の資料はとてもわかりやすいので、ぜひ読んでみてください。
ファイルcomplex-steerable-pyramid.py の関数complex-steerable-pyramid()を呼び出して使います。詳しい実装は各ソースコードや、参考にあるリンクを参照してください。

## 参考
[Steerable Pyramid](https://medium.com/@itberrios6/steerable-pyramids-6bfd4d23c10d)  
[Complex Steerable Pyramid](https://medium.com/@itberrios6/complex-steerable-pyramids-3cf7b99ff9fc)  
[参考にしたソースコード(GitHub)](https://github.com/itberrios/CV_projects/blob/main/pyramids/steerable_pyramids.ipynb)  

## License
MITライセンスに則り、My-Complex-Steerable-Pyramid内のすべてのソースコードの利用や変更、再配布などが可能です。
