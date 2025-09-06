   public class 网络模型图像预处理 {

   public class 数据{
        public static readonly DenseTensor<float>[] 内存双缓数据 =
    {
      new DenseTensor<float>(new float[3*640*640],[1,3,640,640]),
      new DenseTensor<float>(new float[3*640*640],[1,3,640,640]),

    };
    public static int 内存双缓索引 = 0;
   }

    private readonly Scalar 填充颜色 = new Scalar(144, 144, 144);// 填充颜色
   //参照YOLO论坛下的 letterbox
   public  Mat 图像缩放(Mat 导入的图片, OpenCvSharp.Size 目标尺寸)
   {
       if (导入的图片.Empty())
       {
           Console.WriteLine("图片缩放: 图像为空");
           throw new Exception(" ");
       }

        var 缩放比例_恢复 = 0;
        double 缩放比例 = Math.Min((double)目标尺寸.Width / 导入的图片.Cols, (double)目标尺寸.Height / 导入的图片.Rows);
       // Console.WriteLine("得到的缩放比例" + 缩放比例);
       //缩放比例大小反向映射 
       var 缩放比例_恢复 = 1 / 缩放比例;
       // Console.WriteLine("得到的缩放比例" + 公共数据.缩放比例);
        
        
       //计算目标图像的尺寸

       int 目标_图片尺寸宽 = (int)Math.Round(导入的图片.Cols * 缩放比例);
       int 目标_图片尺寸高 = (int)Math.Round(导入的图片.Rows *  缩放比例);
       // Console.WriteLine("构成图片的宽度:" + 目标_图片尺寸宽);
       // Console.WriteLine("构成图片的高度:" + 目标_图片尺寸高);

       Mat 缩放_保存图片 = new Mat();
       Cv2.Resize(导入的图片, 缩放_保存图片, new OpenCvSharp.Size(目标_图片尺寸宽, 目标_图片尺寸高), 0, 0, InterpolationFlags.Linear);

       // 计算与填充区域

       int 填充区域_top = (目标尺寸.Height - 目标_图片尺寸高) / 2;
       int 填充区域_bottom = 目标尺寸.Height - 目标_图片尺寸高 - 填充区域_top;
       int 填充区域_left = (目标尺寸.Width - 目标_图片尺寸宽) / 2;
       int 填充区域_right = 目标尺寸.Width - 目标_图片尺寸宽 - 填充区域_left;
       // Console.WriteLine("填充区域_top:" + 填充区域_top+"填充区域_bottom:"+填充区域_bottom+"填充区域_left:"+填充区域_left+"填充区域_right:"+填充区域_right);

       // 计算填充补偿
       var 填充偏移x = (640-(导入的图片.Cols * 缩放比例))/ 2;
       var 填充偏移y = (640-(导入的图片.Rows * 缩放比例)) / 2;
       // Console.WriteLine("填充偏移x:" + 填充偏移x + "填充偏移y:" + 填充偏移y);

       using Mat 得到的图片 = new Mat();
       Cv2.CopyMakeBorder(缩放_保存图片, 得到的图片, 填充区域_top, 填充区域_bottom, 填充区域_left, 填充区域_right, BorderTypes.Constant, 填充颜色);
        
       return 得到的图片;
        
   }

   public void 图像预处理CvDnn(Mat 导入的图片)
    {

        OpenCvSharp.Size 图片尺寸 = new OpenCvSharp.Size(640, 640);
        bool 是否RGB = true;
        //采用内部拉取数据方式 
            try
            {
        
                Mat 缩放后的图片 = 图像缩放(导入的图片, 图片尺寸);
                // 缩放后的图片.SaveImage("2.png");
                // 使用 BlobFromImage
                var result = CvDnn.BlobFromImage(
                                   image: 缩放后的图片,
                                   scaleFactor: 1.0 / 255.0,
                                   swapRB: 是否RGB,
                                   crop: false,
                                   mean: new Scalar()

                );
                int 原索引 = 数据.内存双缓索引 ^ 1;

                if (MemoryMarshal.TryGetArray(数据.内存双缓数据[原索引].Buffer, out ArraySegment<float> segment))
                {
                    Marshal.Copy(result.Data, segment.Array!, 0, segment.Count);
                }
                else
                {
                    Console.WriteLine("无法获取缓冲区的内存位置");
                }
                //直接对CvDnn.BlobFromImage 得到的Mat进行拷贝，在这里建议使用指针以及手动管理内存，多余的操作只会增加图像处理时间。
                //详情请问ai

                缩放后的图片.Dispose();
                result.Dispose();
                Console.WriteLine($"图像已处理{数据.计数}.png");
                _预处理就绪信号量.Release();
        
            catch (Exception error)
            {
                string IN = "图像预处理失败，原因：" + error.Message;

                MessageBox.Show("图像预处理错误", "error", MessageBoxButton.OKCancel);
                Console.WriteLine(IN);
                break;


            }

        }

    }

       // //图像预处理 -> 归一， HWC->CHW ,
    public void 图像预处理OPENCV(Mat WGC得到的图像)
   {

       try
       {
           // 创建新的图片存放对象
           Mat 处理后的图像 = new Mat();
           // 图像缩放
           Mat 缩放后的图片 = 图像缩放(WGC得到的图像, new OpenCvSharp.Size(640, 640));

           // rgb归一
           缩放后的图片.ConvertTo(处理后的图像, MatType.CV_32FC3, 1.0 / 255.0);
          Console.WriteLine("ConvertTo完成,已将图像255归一");

           Mat[] chwMats = Cv2.Split(处理后的图像);

           // 分别读取每个通道的数据
           chwMats[0].GetArray(out float[] rData); // R
           chwMats[1].GetArray(out float[] gData); // G
           chwMats[2].GetArray(out float[] bData); // B


           float[] chwArray = new float[3 * 处理后的图像.Height * 处理后的图像.Width];
           Array.Copy(rData, 0, chwArray, 0 * 处理后的图像.Height * 处理后的图像.Width, 处理后的图像.Height * 处理后的图像.Width);
           Array.Copy(gData, 0, chwArray, 1 * 处理后的图像.Height * 处理后的图像.Width, 处理后的图像.Height * 处理后的图像.Width);
           Array.Copy(bData, 0, chwArray, 2 * 处理后的图像.Height * 处理后的图像.Width, 处理后的图像.Height * 处理后的图像.Width);
           foreach (var m in chwMats) m.Dispose(); //释放内存

          Console.WriteLine("通道转换完成");

           lock (one_内存锁)
           { //采用异步写入，如有需要可以返回 ，但是需要using Mat

               处理后数据 = new DenseTensor<float>(chwArray, new int[] { 1, 3, 处理后的图像.Height, 处理后的图像.Width });
           }
    
           //对开始创建的控件进行释放
           处理后的图像.Dispose();
           缩放后的图片.Dispose();

       }
       catch (Exception error)
       {
           string IN = "图像预处理失败，原因：" + error.Message;
           Console.WriteLine(IN);
           MessageBox.Show("图像预处理错误", "error", MessageBoxButtons.OKCancel);
           throw new Exception("");

       }
      Console.WriteLine("图像预处理完成");
       图像IF = true;
      
   }
}


public class 模型加载示例 {
    
    
    // 模型设置部分
    public static readonly SessionOptions 模型设置 = BuildOnce();

     private static SessionOptions BuildOnce()
    { 
        var opt = SessionOptions.MakeSessionOptionWithCudaProvider(0);

        var trt = new OrtTensorRTProviderOptions();
        // trt.UpdateOptions(new Dictionary<string, string>
        // {
        //     { "trt_fp16_enable", "1" }
        // });
        // opt.AppendExecutionProvider_Tensorrt(trt);

        opt.ExecutionMode = ExecutionMode.ORT_PARALLEL;
        opt.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        opt.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        opt.AddSessionConfigEntry("session.use_device_allocator_for_initializers", "1");
        opt.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
        return opt;
    }
    
     public void 模型加载(){
         string 模型path = "yolov4.onnx";
         using var YOLO模型 = new InferenceSession(模型path, 模型设置);
         Console.WriteLine("YOLO:模型加载完成");

        try
       {

           using var 推理结果 = YOLO模型.Run(new[] { NamedOnnxValue.CreateFromTensor("images", 处理后数据!) });
           处理后数据 = null;
         Console.WriteLine("YOLO:推理完成");
         Console.WriteLine("YOLO:数据开始筛选");
           var 待分选data = 推理结果.First(v => v.Name == "output0").AsTensor<float>();

           var flat = 待分选data.ToDenseTensor().Buffer.Span;
           var 检测出的框 = new List<OpenCvSharp.Rect>();
           var 分数列表 = new List<float>();
           int 数据步长 = 待分选data.Dimensions[1];

           推理结果.Dispose();

           for (int i = 0; i < 待分选data.Dimensions[2]; i++) //单一类别
           {
               float x = flat[0 * 待分选data.Dimensions[2] + i];
               float y = flat[1 * 待分选data.Dimensions[2] + i];
               float w = flat[2 * 待分选data.Dimensions[2] + i];
               float h = flat[3 * 待分选data.Dimensions[2] + i];
               float 当前数据置信度 = flat[4 * 待分选data.Dimensions[2] + i];

               if (当前数据置信度 > 0.5f) //这里的0.5是目标置信度
               {
                   int x1 = (int)(x - w / 2);
                   int y1 = (int)(y - h / 2);

                   检测出的框.Add(new OpenCvSharp.Rect(
                       x1,
                       y1,
                       (int)w,
                       (int)h
                   ));
                   分数列表.Add(当前数据置信度);
               }
           }

           CvDnn.NMSBoxes(检测出的框, 分数列表, 0.5f, 0.5f, out int[] 得到的NMS列表);
         Console.WriteLine("YOLO:数据筛选完成");
         //数据还没有复原至原图片

       }
       catch (Exception error)
       {
           string IN = "YOLO:推理失败，原因：" + error.Message;
           Console.WriteLine(IN);
           throw new Exception();
       }


//注意， 使用的Onnx模型时候需要对当前 cuda cudnn trt等进行适配否则将会报错
// 使用trt 时候需要对模型的输入部分 进行量化匹配 如 fp32 fp16 int8等 不然转为trt效果只有一半，关键点还是数据量的处理
// 还需要设置 模型输入口，输出口
}
}