using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Management;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SmartCanvasCropperLauncher
{
    internal static class AppConfig
    {
        internal const string ProductVersion = "1.8.1";
        internal const string RuntimeSchema = "portable-py3119-cuda271-dml241-cpu271-ultralytics8437-r5";
        internal const string PrimaryDetectorSha256 = "946E70DE8D586E84F1BB98CAAB8C094E3A8CF501E4807888166299439A67CBA3";
        internal const string RecoveryDetectorSha256 = "9915A40C0AF95E45B38FAB9A20678E7A089914959205239A713A34958599455B";
        internal const string SegmenterSha256 = "6DBB90523A35330FEDD7F1D3DFC66F995213D81B29A5CA8108DBCDD4E37D6C2F";

        internal static readonly string BaseDirectory = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
        internal static readonly string AppDirectory = Path.Combine(BaseDirectory, "app");
        internal static readonly string LocalRoot = ResolveLocalRoot();
        internal static readonly string LogDirectory = Path.Combine(LocalRoot, "logs");
        internal static readonly string PythonBaseDirectory = Path.Combine(BaseDirectory, "python-base");

        private static string ResolveLocalRoot()
        {
            string testRoot = Environment.GetEnvironmentVariable("SMART_CROPPER_LOCAL_ROOT");
            if (!string.IsNullOrWhiteSpace(testRoot))
                return Path.GetFullPath(Environment.ExpandEnvironmentVariables(testRoot.Trim()));
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "SmartCanvasCropper",
                "v1.8");
        }
    }

    internal sealed class HardwareInfo
    {
        internal bool Is64BitOperatingSystem;
        internal bool HasNvidia;
        internal bool HasAmd;
        internal bool HasIntelGpu;
        internal List<string> AdapterNames = new List<string>();

        internal string PreferredMode
        {
            get
            {
                if (HasNvidia)
                    return "nvidia";
                if (HasAmd || HasIntelGpu)
                    return "directml";
                return "cpu";
            }
        }

        internal string DisplayName
        {
            get
            {
                if (AdapterNames.Count == 0)
                    return "未识别到独立显卡";
                return string.Join(" / ", AdapterNames.Distinct(StringComparer.OrdinalIgnoreCase).ToArray());
            }
        }
    }

    internal static class HardwareDetector
    {
        internal static HardwareInfo Detect()
        {
            HardwareInfo result = new HardwareInfo();
            result.Is64BitOperatingSystem = Environment.Is64BitOperatingSystem;

            string testOverride = Environment.GetEnvironmentVariable("SMART_CROPPER_TEST_GPU");
            if (!string.IsNullOrWhiteSpace(testOverride))
            {
                string normalized = testOverride.Trim().ToUpperInvariant();
                if (normalized == "NVIDIA")
                {
                    result.AdapterNames.Add("NVIDIA 测试显卡");
                    result.HasNvidia = true;
                }
                else if (normalized == "AMD")
                {
                    result.AdapterNames.Add("AMD Radeon 测试显卡");
                    result.HasAmd = true;
                }
                else if (normalized == "INTEL")
                {
                    result.AdapterNames.Add("Intel Arc 测试显卡");
                    result.HasIntelGpu = true;
                }
                else
                {
                    result.AdapterNames.Add("通用 CPU 测试环境");
                }
                return result;
            }

            try
            {
                using (ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT Name FROM Win32_VideoController"))
                using (ManagementObjectCollection objects = searcher.Get())
                {
                    foreach (ManagementObject item in objects)
                    {
                        string name = Convert.ToString(item["Name"], CultureInfo.InvariantCulture);
                        if (string.IsNullOrWhiteSpace(name))
                            continue;
                        result.AdapterNames.Add(name.Trim());
                        if (name.IndexOf("NVIDIA", StringComparison.OrdinalIgnoreCase) >= 0)
                            result.HasNvidia = true;
                        if (name.IndexOf("AMD", StringComparison.OrdinalIgnoreCase) >= 0 ||
                            name.IndexOf("Radeon", StringComparison.OrdinalIgnoreCase) >= 0)
                            result.HasAmd = true;
                        if (name.IndexOf("Intel", StringComparison.OrdinalIgnoreCase) >= 0 &&
                            (name.IndexOf("Arc", StringComparison.OrdinalIgnoreCase) >= 0 ||
                             name.IndexOf("Iris", StringComparison.OrdinalIgnoreCase) >= 0 ||
                             name.IndexOf("UHD", StringComparison.OrdinalIgnoreCase) >= 0 ||
                             name.IndexOf("HD Graphics", StringComparison.OrdinalIgnoreCase) >= 0))
                            result.HasIntelGpu = true;
                    }
                }
            }
            catch
            {
                // A locked-down customer PC can deny WMI. The nvidia-smi probe below
                // still identifies NVIDIA installations without changing system state.
            }

            if (!result.HasNvidia)
            {
                string nvidiaName = ProbeNvidiaSmi();
                if (!string.IsNullOrWhiteSpace(nvidiaName))
                {
                    result.HasNvidia = true;
                    result.AdapterNames.Add(nvidiaName);
                }
            }
            return result;
        }

        private static string ProbeNvidiaSmi()
        {
            try
            {
                ProcessStartInfo info = new ProcessStartInfo("nvidia-smi.exe", "--query-gpu=name --format=csv,noheader");
                info.UseShellExecute = false;
                info.CreateNoWindow = true;
                info.RedirectStandardOutput = true;
                info.RedirectStandardError = true;
                using (Process process = Process.Start(info))
                {
                    if (!process.WaitForExit(5000))
                    {
                        try { process.Kill(); } catch { }
                        return null;
                    }
                    if (process.ExitCode != 0)
                        return null;
                    string first = process.StandardOutput.ReadToEnd()
                        .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                        .FirstOrDefault();
                    return string.IsNullOrWhiteSpace(first) ? null : first.Trim();
                }
            }
            catch
            {
                return null;
            }
        }
    }

    internal static class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            ServicePointManager.SecurityProtocol = (SecurityProtocolType)3072;
            string diagnosticsPath = GetOptionValue(args, "--diagnostics-out");
            if (!string.IsNullOrWhiteSpace(diagnosticsPath))
            {
                try
                {
                    WriteDiagnostics(Path.GetFullPath(diagnosticsPath));
                    Environment.ExitCode = 0;
                }
                catch
                {
                    Environment.ExitCode = 2;
                }
                return;
            }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new LauncherForm());
        }

        private static string GetOptionValue(string[] args, string option)
        {
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (string.Equals(args[i], option, StringComparison.OrdinalIgnoreCase))
                    return args[i + 1];
            }
            return null;
        }

        private static void WriteDiagnostics(string path)
        {
            HardwareInfo hardware = HardwareDetector.Detect();
            string runtimeRoot = Path.Combine(AppConfig.LocalRoot, "runtime-" + hardware.PreferredMode);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            StringBuilder report = new StringBuilder();
            report.AppendLine("SmartCanvasCropper=" + AppConfig.ProductVersion);
            report.AppendLine("OS64=" + hardware.Is64BitOperatingSystem);
            report.AppendLine("GPU=" + hardware.DisplayName);
            report.AppendLine("PreferredMode=" + hardware.PreferredMode);
            report.AppendLine("RuntimeRoot=" + runtimeRoot);
            report.AppendLine("PythonBase=" + AppConfig.PythonBaseDirectory);
            report.AppendLine("PythonDeployment=BundledPortableCopy");
            report.AppendLine("BackendPlan=NVIDIA:CUDA|AMD:DirectML|Intel:DirectML|Fallback:DownloadedCPU");
            report.AppendLine("PackageIndex=" + (hardware.PreferredMode == "nvidia"
                ? "https://download.pytorch.org/whl/cu126"
                : hardware.PreferredMode == "directml" ? "https://pypi.org/simple" : "https://download.pytorch.org/whl/cpu"));
            report.AppendLine("PipConfigIsolation=nul");
            report.AppendLine("PipProgress=package|downloaded|total|speed");
            report.AppendLine("SourceConnectTimeoutSeconds=30");
            report.AppendLine("SourceStallTimeoutSeconds=120");
            report.AppendLine("CommonSources=https://pypi.org/simple|https://mirrors.aliyun.com/pypi/simple|https://pypi.tuna.tsinghua.edu.cn/simple");
            report.AppendLine("TorchMirrorFallback=NJU|SJTUG");
            report.AppendLine("UnicodePathProbe=" + UnicodePathProbe());
            File.WriteAllText(path, report.ToString(), new UTF8Encoding(true));
        }

        internal static string UnicodePathProbe()
        {
            string root = Path.Combine(
                Path.GetTempPath(),
                "智能裁图_中文路径检测_" + Guid.NewGuid().ToString("N"));
            string file = Path.Combine(root, "读写验证.txt");
            try
            {
                Directory.CreateDirectory(root);
                File.WriteAllText(file, "中文路径正常", new UTF8Encoding(false));
                string value = File.ReadAllText(file, Encoding.UTF8);
                try { File.Delete(file); } catch { }
                try { Directory.Delete(root, false); } catch { }
                return value == "中文路径正常" ? "PASS" : "FAIL";
            }
            catch (Exception ex)
            {
                return "FAIL:" + ex.GetType().Name;
            }
        }
    }

    internal sealed class LauncherForm : Form
    {
        private readonly Label titleLabel = new Label();
        private readonly Label hardwareLabel = new Label();
        private readonly Label statusLabel = new Label();
        private readonly Label downloadDetailLabel = new Label();
        private readonly ProgressBar progressBar = new ProgressBar();
        private readonly TextBox logBox = new TextBox();
        private readonly Button retryButton = new Button();
        private readonly Button cpuButton = new Button();
        private readonly CancellationTokenSource cancellation = new CancellationTokenSource();
        private readonly object logLock = new object();
        private HardwareInfo hardware;
        private string logFile;
        private bool busy;

        internal LauncherForm()
        {
            Text = "智能裁图工具 " + AppConfig.ProductVersion;
            StartPosition = FormStartPosition.CenterScreen;
            ClientSize = new Size(720, 550);
            MinimumSize = new Size(640, 500);
            Font = new Font("Microsoft YaHei UI", 9F);
            BackColor = Color.FromArgb(246, 248, 251);

            titleLabel.Text = "智能裁图工具 · NVIDIA / AMD / Intel / CPU 自动适配版";
            titleLabel.Font = new Font("Microsoft YaHei UI", 17F, FontStyle.Bold);
            titleLabel.AutoSize = true;
            titleLabel.Location = new Point(28, 24);

            hardwareLabel.AutoSize = false;
            hardwareLabel.Location = new Point(31, 69);
            hardwareLabel.Size = new Size(655, 44);
            hardwareLabel.ForeColor = Color.FromArgb(55, 65, 81);

            statusLabel.AutoSize = false;
            statusLabel.Location = new Point(31, 120);
            statusLabel.Size = new Size(655, 24);
            statusLabel.Font = new Font("Microsoft YaHei UI", 10F, FontStyle.Bold);
            statusLabel.Text = "正在检查本机环境…";

            downloadDetailLabel.AutoSize = false;
            downloadDetailLabel.Location = new Point(31, 146);
            downloadDetailLabel.Size = new Size(655, 36);
            downloadDetailLabel.ForeColor = Color.FromArgb(55, 65, 81);
            downloadDetailLabel.Text = "等待硬件检测";

            progressBar.Location = new Point(34, 185);
            progressBar.Size = new Size(652, 19);
            progressBar.Style = ProgressBarStyle.Marquee;
            progressBar.MarqueeAnimationSpeed = 22;

            logBox.Location = new Point(34, 218);
            logBox.Size = new Size(652, 270);
            logBox.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
            logBox.Multiline = true;
            logBox.ScrollBars = ScrollBars.Vertical;
            logBox.ReadOnly = true;
            logBox.BackColor = Color.White;
            logBox.Font = new Font("Consolas", 9F);

            retryButton.Text = "重试";
            retryButton.Size = new Size(92, 32);
            retryButton.Location = new Point(594, 504);
            retryButton.Anchor = AnchorStyles.Bottom | AnchorStyles.Right;
            retryButton.Visible = false;
            retryButton.Click += async delegate { await BeginSetupAsync(); };

            cpuButton.Text = "CPU 兼容模式";
            cpuButton.Size = new Size(142, 32);
            cpuButton.Location = new Point(438, 504);
            cpuButton.Anchor = AnchorStyles.Bottom | AnchorStyles.Right;
            cpuButton.Visible = false;

            Controls.Add(titleLabel);
            Controls.Add(hardwareLabel);
            Controls.Add(statusLabel);
            Controls.Add(downloadDetailLabel);
            Controls.Add(progressBar);
            Controls.Add(logBox);
            Controls.Add(retryButton);
            Controls.Add(cpuButton);

            Shown += async delegate { await BeginSetupAsync(); };
            FormClosing += delegate { cancellation.Cancel(); };
        }

        private async Task BeginSetupAsync()
        {
            if (busy)
                return;
            busy = true;
            retryButton.Visible = false;
            logBox.Clear();
            progressBar.Style = ProgressBarStyle.Marquee;
            progressBar.MarqueeAnimationSpeed = 22;
            downloadDetailLabel.Text = "正在读取硬件信息…";

            Exception setupFailure = null;
            bool applicationLaunched = false;

            try
            {
                Directory.CreateDirectory(AppConfig.LogDirectory);
                logFile = Path.Combine(AppConfig.LogDirectory, "launcher-" + DateTime.Now.ToString("yyyyMMdd-HHmmss") + ".log");
                ValidatePayload();
                hardware = HardwareDetector.Detect();
                if (!hardware.Is64BitOperatingSystem)
                    throw new InvalidOperationException("本工具需要 64 位 Windows 10 或 Windows 11。");

                string plan = hardware.PreferredMode == "nvidia"
                    ? "NVIDIA 显卡使用 CUDA；CUDA 不可用时尝试 DirectML，最后下载 CPU 兼容环境。"
                    : hardware.PreferredMode == "directml"
                        ? "AMD / Intel 显卡使用 DirectML GPU；不可用时下载 CPU 兼容环境。"
                        : "未找到可用 GPU，将下载 CPU 兼容环境。";
                hardwareLabel.Text = "检测到：" + hardware.DisplayName + Environment.NewLine + plan;
                AppendLog("硬件检测：" + hardware.DisplayName);
                AppendLog("后端策略：NVIDIA=CUDA，AMD/Intel=DirectML，最终兜底=按需下载CPU。");
                AppendLog("中文路径基础检测：" + Program.UnicodePathProbe());

                string selectedMode = hardware.PreferredMode;
                List<string> candidates = new List<string>();
                candidates.Add(selectedMode);
                if (selectedMode == "nvidia")
                    candidates.Add("directml");
                if (!candidates.Contains("cpu"))
                    candidates.Add("cpu");

                foreach (string candidate in candidates)
                {
                    Exception backendFailure = await TryEnsureRuntimeAsync(candidate, cancellation.Token);
                    if (backendFailure == null)
                    {
                        selectedMode = candidate;
                        statusLabel.Text = candidate == "nvidia"
                            ? "环境就绪，正在以 NVIDIA CUDA GPU 启动…"
                            : candidate == "directml"
                                ? "环境就绪，正在以 DirectML GPU 启动…"
                                : "环境就绪，正在以 CPU 兼容模式启动…";
                        downloadDetailLabel.Text = "运行环境与模型已通过完整性检查";
                        AppendLog("运行模式：" + (candidate == "nvidia"
                            ? "NVIDIA CUDA GPU" : candidate == "directml" ? "DirectML GPU" : "CPU"));
                        progressBar.Style = ProgressBarStyle.Continuous;
                        progressBar.Value = 100;
                        LaunchApplication(candidate);
                        applicationLaunched = true;
                        break;
                    }

                    AppendLog((candidate == "nvidia" ? "NVIDIA CUDA" : candidate == "directml" ? "DirectML" : "CPU") +
                        " 环境准备未成功：" + backendFailure.Message);
                }

                if (!applicationLaunched)
                    throw new InvalidOperationException("GPU 与 CPU 运行环境均未能完成准备。");
            }
            catch (OperationCanceledException ex)
            {
                setupFailure = ex;
            }
            catch (Exception ex)
            {
                setupFailure = ex;
            }
            finally
            {
                busy = false;
            }

            if (setupFailure != null)
            {
                AppendLog("环境错误：" + setupFailure.Message);
                statusLabel.Text = "环境准备失败";
                downloadDetailLabel.Text = "请查看日志后重试";
                AppendLog("详细日志：" + logFile);
                progressBar.Style = ProgressBarStyle.Continuous;
                progressBar.Value = 0;
                retryButton.Visible = true;
                MessageBox.Show(this,
                    setupFailure.Message + Environment.NewLine + Environment.NewLine + "详细日志：" + logFile,
                    "智能裁图工具 - 启动失败",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
                return;
            }

            if (applicationLaunched)
            {
                await Task.Delay(700);
                Close();
            }
        }

        private async Task<Exception> TryEnsureRuntimeAsync(string mode, CancellationToken token)
        {
            try
            {
                await EnsureRuntimeAsync(mode, token);
                return null;
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                return ex;
            }
        }

        private void ValidatePayload()
        {
            string source = Path.Combine(AppConfig.AppDirectory, "smart_canvas_cropper.py");
            string primaryDetector = Path.Combine(AppConfig.AppDirectory, "yolov8l-worldv2-canvas.pt");
            string recoveryDetector = Path.Combine(AppConfig.AppDirectory, "yolov8s-worldv2-canvas.pt");
            string segmenter = Path.Combine(AppConfig.AppDirectory, "mobile_sam.pt");
            if (!File.Exists(source) || !File.Exists(primaryDetector) ||
                !File.Exists(recoveryDetector) || !File.Exists(segmenter))
                throw new FileNotFoundException("程序文件不完整，请重新解压完整 ZIP 后再运行。");
            if (!File.Exists(Path.Combine(AppConfig.PythonBaseDirectory, "python.exe")) ||
                !Directory.Exists(Path.Combine(AppConfig.PythonBaseDirectory, "Lib", "site-packages", "pip")) ||
                !File.Exists(Path.Combine(AppConfig.PythonBaseDirectory, "DLLs", "_tkinter.pyd")))
                throw new FileNotFoundException("内置 Python 基础运行时不完整，请重新解压完整 ZIP 后再运行。");
            if (!HashMatches(primaryDetector, AppConfig.PrimaryDetectorSha256) ||
                !HashMatches(recoveryDetector, AppConfig.RecoveryDetectorSha256) ||
                !HashMatches(segmenter, AppConfig.SegmenterSha256))
                throw new InvalidDataException("模型权重校验失败，请重新解压完整 ZIP 后再运行。");
        }

        private async Task EnsureRuntimeAsync(string mode, CancellationToken token)
        {
            string runtimeRoot = GetRuntimeRoot(mode);
            string marker = Path.Combine(runtimeRoot, "runtime-ready.txt");
            string python = Path.Combine(runtimeRoot, "python.exe");
            string expectedMarker = AppConfig.RuntimeSchema + "|" + mode;
            if (File.Exists(python) && File.Exists(marker) && File.ReadAllText(marker, Encoding.UTF8).Trim() == expectedMarker)
            {
                statusLabel.Text = "已找到本机运行环境，正在验证推理设备…";
                AppendLog("正在验证已缓存运行环境：" + runtimeRoot);
                try
                {
                    await ValidateRuntimeAsync(mode, python, runtimeRoot, token);
                    AppendLog("已缓存运行环境验证通过。");
                    return;
                }
                catch (Exception ex)
                {
                    AppendLog("已缓存运行环境验证失败，将修复环境：" + ex.Message);
                    try { File.Delete(marker); } catch { }
                }
            }

            Directory.CreateDirectory(runtimeRoot);
            statusLabel.Text = mode == "nvidia"
                ? "首次运行：正在准备 NVIDIA GPU 环境（约需下载 2.5–3 GB）"
                : mode == "directml"
                    ? "首次运行：正在准备 DirectML GPU 环境（约需下载 350–550 MB）"
                    : "首次运行：正在准备 CPU 兼容环境（约需下载 250–450 MB）";
            AppendLog("运行环境目录：" + runtimeRoot);

            await EnsurePythonAsync(runtimeRoot, token);
            token.ThrowIfCancellationRequested();

            python = Path.Combine(runtimeRoot, "python.exe");
            await EnsurePipProgressSupportAsync(python, runtimeRoot, token);

            string[] commonIndexes = new[]
            {
                "https://pypi.org/simple",
                "https://mirrors.aliyun.com/pypi/simple",
                "https://pypi.tuna.tsinghua.edu.cn/simple"
            };

            if (mode == "nvidia")
            {
                string[] torchIndexes = new[]
                {
                    "https://download.pytorch.org/whl/cu126",
                    "https://mirrors.nju.edu.cn/pytorch/whl/cu126",
                    "https://mirror.sjtu.edu.cn/pytorch-wheels/cu126"
                };
                await InstallWithSourceFallbackAsync(
                    python,
                    "torch==2.7.1 torchvision==0.22.1",
                    torchIndexes,
                    runtimeRoot,
                    "正在安装 NVIDIA CUDA 推理组件",
                    false,
                    token);
            }
            else if (mode == "directml")
            {
                await InstallWithSourceFallbackAsync(
                    python,
                    "torch-directml==0.2.5.dev240914",
                    commonIndexes,
                    runtimeRoot,
                    "正在安装 AMD / Intel DirectML 推理组件",
                    true,
                    token);
            }
            else
            {
                string[] cpuIndexes = new[]
                {
                    "https://download.pytorch.org/whl/cpu",
                    "https://mirrors.nju.edu.cn/pytorch/whl/cpu",
                    "https://mirror.sjtu.edu.cn/pytorch-wheels/cpu"
                };
                await InstallWithSourceFallbackAsync(
                    python,
                    "torch==2.7.1 torchvision==0.22.1",
                    cpuIndexes,
                    runtimeRoot,
                    "正在安装 CPU 推理组件",
                    false,
                    token);
            }

            await InstallWithSourceFallbackAsync(
                python,
                "ultralytics==8.4.37 opencv-python==4.11.0.86 pillow==11.3.0 tqdm==4.67.1",
                commonIndexes,
                runtimeRoot,
                "正在安装图像处理组件",
                false,
                token);

            await ValidateRuntimeAsync(mode, python, runtimeRoot, token);
            File.WriteAllText(marker, expectedMarker, new UTF8Encoding(false));
            AppendLog("运行环境安装完成。");
        }

        private async Task EnsurePipProgressSupportAsync(string python, string runtimeRoot, CancellationToken token)
        {
            string[] indexes = new[]
            {
                "https://pypi.org/simple",
                "https://mirrors.aliyun.com/pypi/simple",
                "https://pypi.tuna.tsinghua.edu.cn/simple"
            };
            await InstallWithSourceFallbackAsync(
                python,
                "pip==25.2",
                indexes,
                runtimeRoot,
                "正在安装下载进度组件",
                false,
                token,
                false);
        }

        private async Task EnsurePythonAsync(string runtimeRoot, CancellationToken token)
        {
            string python = Path.Combine(runtimeRoot, "python.exe");
            if (File.Exists(python))
            {
                AppendLog("已找到独立 Python 运行时。");
                return;
            }

            token.ThrowIfCancellationRequested();
            statusLabel.Text = "正在部署包内 Python 3.11 基础运行时…";
            downloadDetailLabel.Text = "本地复制，无需修改系统 Python";
            progressBar.Style = ProgressBarStyle.Marquee;
            progressBar.MarqueeAnimationSpeed = 22;
            AppendLog("从包内基础运行时部署 Python：" + AppConfig.PythonBaseDirectory);
            CopyDirectory(AppConfig.PythonBaseDirectory, runtimeRoot, token);
            if (!File.Exists(python))
                throw new InvalidOperationException("包内 Python 基础运行时部署失败。");

            await RunCheckedAsync(
                python,
                "-c " + Quote("import sys,pip,tkinter; print(sys.version.split()[0]); print('pip='+pip.__version__); print('tk='+str(tkinter.TkVersion))"),
                runtimeRoot,
                "正在检查包内 Python 基础运行时…",
                token);
            AppendLog("包内 Python 3.11 基础运行时部署完成。");
        }

        private static void CopyDirectory(string sourceRoot, string destinationRoot, CancellationToken token)
        {
            Directory.CreateDirectory(destinationRoot);
            foreach (string directory in Directory.GetDirectories(sourceRoot, "*", SearchOption.AllDirectories))
            {
                token.ThrowIfCancellationRequested();
                string relative = directory.Substring(sourceRoot.Length).TrimStart(Path.DirectorySeparatorChar);
                Directory.CreateDirectory(Path.Combine(destinationRoot, relative));
            }
            foreach (string sourceFile in Directory.GetFiles(sourceRoot, "*", SearchOption.AllDirectories))
            {
                token.ThrowIfCancellationRequested();
                string relative = sourceFile.Substring(sourceRoot.Length).TrimStart(Path.DirectorySeparatorChar);
                string destination = Path.Combine(destinationRoot, relative);
                Directory.CreateDirectory(Path.GetDirectoryName(destination));
                File.Copy(sourceFile, destination, true);
            }
        }

        private async Task ValidateRuntimeAsync(string mode, string python, string runtimeRoot, CancellationToken token)
        {
            string code;
            string modelProbe =
                "import smart_canvas_cropper as app; from ultralytics import YOLO,SAM; sys.stderr=None; " +
                "primary=YOLO(str(app.WEIGHTS)); recovery=YOLO(str(app.RECOVERY_WEIGHTS)); segmenter=SAM(str(app.SEGMENT_WEIGHTS)); " +
                "assert getattr(primary.model,'txt_feats',None) is not None; " +
                "assert getattr(recovery.model,'txt_feats',None) is not None; " +
                "print('primary_features='+str(tuple(primary.model.txt_feats.shape))); " +
                "print('recovery_features='+str(tuple(recovery.model.txt_feats.shape))); ";
            if (mode == "nvidia")
            {
                code = "import os,sys,torch,cv2,PIL,ultralytics; " +
                    "os.environ['SMART_CROPPER_RUNTIME_MODE']='nvidia'; " + modelProbe +
                    "print('python='+sys.version.split()[0]); print('torch='+torch.__version__); " +
                    "print('cuda='+str(torch.cuda.is_available())); print('opencv='+cv2.__version__); " +
                    "print('ultralytics='+ultralytics.__version__); " +
                    "raise SystemExit(0 if torch.cuda.is_available() else 3)";
            }
            else if (mode == "directml")
            {
                code = "import os,sys,torch,cv2,PIL,ultralytics; " +
                    "os.environ['SMART_CROPPER_RUNTIME_MODE']='directml'; " +
                    modelProbe + "d,label=app.choose_compute_device(); " +
                    "v=torch.ones(1).to(d).cpu(); print('python='+sys.version.split()[0]); " +
                    "print('torch='+torch.__version__); print('directml='+str(d)); print('label='+label); " +
                    "print('opencv='+cv2.__version__); print('ultralytics='+ultralytics.__version__); " +
                    "raise SystemExit(0 if str(d).startswith('privateuseone') and float(v[0])==1.0 else 4)";
            }
            else
            {
                code = "import os,sys,torch,cv2,PIL,ultralytics; " +
                    "os.environ['SMART_CROPPER_RUNTIME_MODE']='cpu'; " + modelProbe +
                    "d,label=app.choose_compute_device(); v=torch.ones(1).to(d); " +
                    "print('python='+sys.version.split()[0]); print('torch='+torch.__version__); " +
                    "print('device='+str(d)); print('label='+label); print('opencv='+cv2.__version__); " +
                    "print('ultralytics='+ultralytics.__version__); " +
                    "raise SystemExit(0 if str(d)=='cpu' and float(v[0])==1.0 else 5)";
            }
            ProcessResult result = await RunProcessAsync(python, "-c " + Quote(code), runtimeRoot, token);
            AppendProcessOutput(result);
            if (result.ExitCode != 0)
            {
                if (mode == "nvidia" && result.ExitCode == 3)
                    throw new InvalidOperationException("当前 NVIDIA 驱动未能启用 CUDA，将继续尝试 DirectML GPU。");
                if (mode == "directml" && result.ExitCode == 4)
                    throw new InvalidOperationException("当前显卡或驱动未能启用 DirectML GPU。");
                if (mode == "cpu" && result.ExitCode == 5)
                    throw new InvalidOperationException("当前系统未能启用 CPU 推理环境。");
                throw new InvalidOperationException("运行环境完整性校验失败（代码 " + result.ExitCode + "）。");
            }
        }

        private async Task RunCheckedAsync(string executable, string arguments, string workingDirectory, string status, CancellationToken token)
        {
            statusLabel.Text = status;
            AppendLog(status);
            ProcessResult result = await RunProcessAsync(executable, arguments, workingDirectory, token);
            AppendProcessOutput(result);
            if (result.ExitCode != 0)
                throw new InvalidOperationException(status.TrimEnd('…') + "失败（代码 " + result.ExitCode + "）。");
        }

        private async Task InstallWithSourceFallbackAsync(
            string python,
            string packages,
            string[] indexes,
            string runtimeRoot,
            string status,
            bool preRelease,
            CancellationToken token,
            bool rawProgress = true)
        {
            ProcessResult lastResult = null;
            for (int index = 0; index < indexes.Length; index++)
            {
                token.ThrowIfCancellationRequested();
                string source = indexes[index];
                statusLabel.Text = status + "…（源 " + (index + 1) + "/" + indexes.Length + "）";
                AppendLog(status + "，使用依赖源：" + source);
                string arguments =
                    "-m pip --isolated install --disable-pip-version-check --no-cache-dir " +
                    (preRelease ? "--pre " : string.Empty) +
                    "--progress-bar " + (rawProgress ? "raw" : "off") + " --only-binary=:all: " +
                    "--no-warn-script-location --retries 1 --timeout 30 " +
                    packages + " --index-url " + source;
                lastResult = await RunPipProcessAsync(
                    python, arguments, runtimeRoot, status, source, index + 1, indexes.Length, token);
                if (lastResult.ExitCode == 0)
                {
                    AppendLog(status + "完成。");
                    return;
                }
                AppendLog(lastResult.TimedOut
                    ? "当前依赖源 120 秒无下载进度，已停止并切换下一个来源。"
                    : "当前依赖源安装失败，正在尝试下一个来源。");
            }

            int exitCode = lastResult == null ? -1 : lastResult.ExitCode;
            throw new InvalidOperationException(status + "失败，三个依赖源均不可用（最后代码 " + exitCode + "）。");
        }

        private async Task<ProcessResult> RunPipProcessAsync(
            string executable,
            string arguments,
            string workingDirectory,
            string stage,
            string source,
            int sourceIndex,
            int sourceCount,
            CancellationToken token)
        {
            ProcessStartInfo info = new ProcessStartInfo(executable, arguments);
            info.WorkingDirectory = workingDirectory;
            info.UseShellExecute = false;
            info.CreateNoWindow = true;
            info.RedirectStandardOutput = true;
            info.RedirectStandardError = true;
            info.StandardOutputEncoding = Encoding.UTF8;
            info.StandardErrorEncoding = Encoding.UTF8;
            ConfigureEnvironment(info, AppConfig.AppDirectory);
            info.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";

            PipDisplayState display = new PipDisplayState(stage, source, sourceIndex, sourceCount);
            StringBuilder output = new StringBuilder();
            StringBuilder error = new StringBuilder();
            bool timedOut = false;

            SetPipDisplay(display, "正在连接依赖源");
            LogRaw("COMMAND: " + executable + " " + arguments);

            using (Process process = new Process())
            {
                process.StartInfo = info;
                process.OutputDataReceived += delegate(object sender, DataReceivedEventArgs args)
                {
                    if (args.Data == null)
                        return;
                    lock (output) { output.AppendLine(args.Data); }
                    HandlePipLine(display, args.Data);
                };
                process.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs args)
                {
                    if (args.Data == null)
                        return;
                    lock (error) { error.AppendLine(args.Data); }
                    HandlePipLine(display, args.Data);
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                Task waitTask = Task.Run(delegate { process.WaitForExit(); });
                using (token.Register(delegate
                {
                    try { if (!process.HasExited) process.Kill(); } catch { }
                }))
                {
                    while (!waitTask.IsCompleted)
                    {
                        token.ThrowIfCancellationRequested();
                        DateTime lastActivity;
                        bool networkPhase;
                        lock (display.Sync)
                        {
                            lastActivity = display.LastActivityUtc;
                            networkPhase = display.NetworkPhase;
                        }
                        TimeSpan idle = DateTime.UtcNow - lastActivity;
                        if (networkPhase && idle.TotalSeconds >= 120)
                        {
                            timedOut = true;
                            try { if (!process.HasExited) process.Kill(); } catch { }
                            break;
                        }

                        if (networkPhase)
                            SetPipCountdown(display, Math.Max(0, 120 - (int)idle.TotalSeconds));
                        await Task.Delay(1000, token);
                    }
                    await waitTask;
                }
                process.WaitForExit();

                string outputText;
                string errorText;
                lock (output) { outputText = output.ToString(); }
                lock (error) { errorText = error.ToString(); }
                LogRaw(outputText);
                LogRaw(errorText);
                int exitCode = process.ExitCode;
                return new ProcessResult(exitCode, outputText, errorText, timedOut);
            }
        }

        private void HandlePipLine(PipDisplayState display, string line)
        {
            if (string.IsNullOrWhiteSpace(line))
                return;

            string trimmed = line.Trim();
            bool isProgress = false;
            lock (display.Sync)
            {
                display.LastActivityUtc = DateTime.UtcNow;
                if (trimmed.StartsWith("Collecting ", StringComparison.OrdinalIgnoreCase))
                {
                    string package = trimmed.Substring("Collecting ".Length).Trim();
                    int marker = package.IndexOf(' ');
                    display.PackageName = marker > 0 ? package.Substring(0, marker) : package;
                }
                else if (trimmed.StartsWith("Downloading ", StringComparison.OrdinalIgnoreCase))
                {
                    string value = trimmed.Substring("Downloading ".Length).Trim();
                    int sizeMarker = value.LastIndexOf(" (", StringComparison.Ordinal);
                    string location = sizeMarker > 0 ? value.Substring(0, sizeMarker) : value;
                    Uri uri;
                    string fileName = Uri.TryCreate(location, UriKind.Absolute, out uri)
                        ? Path.GetFileName(uri.LocalPath)
                        : Path.GetFileName(location.Replace('/', Path.DirectorySeparatorChar));
                    if (!string.IsNullOrWhiteSpace(fileName))
                        display.PackageName = fileName;
                    display.DownloadedBytes = 0;
                    display.TotalBytes = 0;
                    display.DownloadWatch.Restart();
                    display.NetworkPhase = true;
                }
                else if (trimmed.StartsWith("Installing collected packages", StringComparison.OrdinalIgnoreCase))
                {
                    display.NetworkPhase = false;
                    display.Installing = true;
                }

                Match match = Regex.Match(trimmed, @"^Progress\s+(\d+)\s+of\s+(\d+)$", RegexOptions.IgnoreCase);
                if (match.Success)
                {
                    long current;
                    long total;
                    if (long.TryParse(match.Groups[1].Value, out current) &&
                        long.TryParse(match.Groups[2].Value, out total))
                    {
                        display.DownloadedBytes = current;
                        display.TotalBytes = total;
                        display.NetworkPhase = true;
                        isProgress = true;
                    }
                }
            }

            LogRaw(line);
            SetPipDisplay(display, display.Installing ? "正在安装和解包" : "正在下载");
            if (!isProgress)
                AppendLog(line);
        }

        private void SetPipCountdown(PipDisplayState display, int remainingSeconds)
        {
            lock (display.Sync)
            {
                display.TimeoutRemaining = remainingSeconds;
            }
            SetPipDisplay(display, "正在下载");
        }

        private void SetPipDisplay(PipDisplayState display, string phase)
        {
            string package;
            string source;
            int sourceIndex;
            int sourceCount;
            int timeoutRemaining;
            long current;
            long total;
            double speed;
            bool installing;
            lock (display.Sync)
            {
                package = display.PackageName;
                source = display.Source;
                sourceIndex = display.SourceIndex;
                sourceCount = display.SourceCount;
                timeoutRemaining = display.TimeoutRemaining;
                current = display.DownloadedBytes;
                total = display.TotalBytes;
                installing = display.Installing;
                speed = display.DownloadWatch.Elapsed.TotalSeconds > 0
                    ? current / display.DownloadWatch.Elapsed.TotalSeconds
                    : 0;
            }

            Ui(delegate
            {
                statusLabel.Text = display.Stage + "（源 " + sourceIndex + "/" + sourceCount + "）";
                if (installing)
                {
                    downloadDetailLabel.Text = "正在安装和解包：" + package + "（此阶段不计网络超时）";
                    progressBar.Style = ProgressBarStyle.Marquee;
                    progressBar.MarqueeAnimationSpeed = 22;
                }
                else
                {
                    string amount = total > 0
                        ? FormatBytes(current) + " / " + FormatBytes(total)
                        : current > 0 ? FormatBytes(current) : "等待数据";
                    downloadDetailLabel.Text = package + " ｜ " + amount + " ｜ " +
                        FormatBytes((long)speed) + "/s ｜ 无进度超时 " + timeoutRemaining + " 秒";
                    progressBar.Style = ProgressBarStyle.Continuous;
                    progressBar.Value = total > 0
                        ? (int)Math.Max(0, Math.Min(100, current * 100L / total))
                        : 0;
                }
                downloadDetailLabel.Tag = source;
            });
        }

        private async Task<ProcessResult> RunProcessAsync(string executable, string arguments, string workingDirectory, CancellationToken token)
        {
            ProcessStartInfo info = new ProcessStartInfo(executable, arguments);
            info.WorkingDirectory = workingDirectory;
            info.UseShellExecute = false;
            info.CreateNoWindow = true;
            info.RedirectStandardOutput = true;
            info.RedirectStandardError = true;
            info.StandardOutputEncoding = Encoding.UTF8;
            info.StandardErrorEncoding = Encoding.UTF8;
            ConfigureEnvironment(info, AppConfig.AppDirectory);

            using (Process process = new Process())
            {
                process.StartInfo = info;
                process.Start();
                Task<string> outputTask = process.StandardOutput.ReadToEndAsync();
                Task<string> errorTask = process.StandardError.ReadToEndAsync();
                using (token.Register(delegate
                {
                    try { if (!process.HasExited) process.Kill(); } catch { }
                }))
                {
                    await Task.Run(delegate { process.WaitForExit(); }, token);
                }
                string output = await outputTask;
                string error = await errorTask;
                LogRaw("COMMAND: " + executable + " " + arguments);
                LogRaw(output);
                LogRaw(error);
                return new ProcessResult(process.ExitCode, output, error);
            }
        }

        private void LaunchApplication(string mode)
        {
            string runtimeRoot = GetRuntimeRoot(mode);
            string pythonw = Path.Combine(runtimeRoot, "pythonw.exe");
            string script = Path.Combine(AppConfig.AppDirectory, "smart_canvas_cropper.py");
            ProcessStartInfo info = new ProcessStartInfo(pythonw, Quote(script));
            info.WorkingDirectory = AppConfig.AppDirectory;
            info.UseShellExecute = false;
            ConfigureEnvironment(info, AppConfig.AppDirectory);
            info.EnvironmentVariables["SMART_CROPPER_RUNTIME_MODE"] = mode;
            info.EnvironmentVariables["SMART_CROPPER_MODEL_DIR"] = AppConfig.AppDirectory;
            Process.Start(info);
        }

        private static void ConfigureEnvironment(ProcessStartInfo info, string appDirectory)
        {
            info.EnvironmentVariables["PYTHONUTF8"] = "1";
            info.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            info.EnvironmentVariables["SMART_CROPPER_LOG_DIR"] = AppConfig.LogDirectory;
            info.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
            info.EnvironmentVariables["PIP_DISABLE_PIP_VERSION_CHECK"] = "1";
            info.EnvironmentVariables["PIP_NO_COLOR"] = "1";
            info.EnvironmentVariables["PIP_CONFIG_FILE"] = "nul";
            info.EnvironmentVariables["PIP_DEFAULT_TIMEOUT"] = "30";
            info.EnvironmentVariables.Remove("PIP_INDEX_URL");
            info.EnvironmentVariables.Remove("PIP_EXTRA_INDEX_URL");
            info.EnvironmentVariables.Remove("PIP_NO_INDEX");
            info.EnvironmentVariables.Remove("PIP_FIND_LINKS");
            info.EnvironmentVariables["PYTHONPATH"] = appDirectory;
            info.EnvironmentVariables["TORCH_HOME"] = Path.Combine(AppConfig.LocalRoot, "torch-cache");
            info.EnvironmentVariables["MPLCONFIGDIR"] = Path.Combine(AppConfig.LocalRoot, "matplotlib-cache");
            info.EnvironmentVariables["ULTRALYTICS_SETTINGS_DIR"] = Path.Combine(AppConfig.LocalRoot, "ultralytics-settings");
        }

        private void Ui(Action action)
        {
            if (IsDisposed || Disposing)
                return;
            if (InvokeRequired)
            {
                try { BeginInvoke(action); } catch { }
                return;
            }
            action();
        }

        private static string FormatBytes(long bytes)
        {
            if (bytes < 0)
                bytes = 0;
            string[] units = new[] { "B", "KB", "MB", "GB" };
            double value = bytes;
            int unit = 0;
            while (value >= 1024 && unit < units.Length - 1)
            {
                value /= 1024;
                unit++;
            }
            return value.ToString(value >= 100 || unit == 0 ? "0" : "0.0", CultureInfo.InvariantCulture) + " " + units[unit];
        }

        private static string GetRuntimeRoot(string mode)
        {
            return Path.Combine(AppConfig.LocalRoot, "runtime-" + mode);
        }

        private static string Quote(string value)
        {
            return "\"" + value.Replace("\"", "\\\"") + "\"";
        }

        private static bool HashMatches(string path, string expected)
        {
            try
            {
                using (FileStream stream = File.OpenRead(path))
                using (SHA256 sha = SHA256.Create())
                {
                    string actual = BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", string.Empty);
                    return string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase);
                }
            }
            catch
            {
                return false;
            }
        }

        private void AppendProcessOutput(ProcessResult result)
        {
            string text = (result.Output + Environment.NewLine + result.Error).Trim();
            if (text.Length == 0)
                return;
            if (text.Length > 3500)
                text = text.Substring(text.Length - 3500);
            AppendLog(text);
        }

        private void AppendLog(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return;
            if (InvokeRequired)
            {
                Ui(delegate { AppendLog(text); });
                return;
            }
            string line = "[" + DateTime.Now.ToString("HH:mm:ss") + "] " + text.Trim();
            logBox.AppendText(line + Environment.NewLine);
            logBox.SelectionStart = logBox.TextLength;
            logBox.ScrollToCaret();
            LogRaw(line);
        }

        private void LogRaw(string text)
        {
            if (string.IsNullOrWhiteSpace(logFile) || string.IsNullOrWhiteSpace(text))
                return;
            lock (logLock)
            {
                File.AppendAllText(logFile, text.TrimEnd() + Environment.NewLine, new UTF8Encoding(true));
            }
        }

        private sealed class PipDisplayState
        {
            internal readonly object Sync = new object();
            internal readonly string Stage;
            internal readonly string Source;
            internal readonly int SourceIndex;
            internal readonly int SourceCount;
            internal readonly Stopwatch DownloadWatch = Stopwatch.StartNew();
            internal string PackageName = "正在解析依赖";
            internal long DownloadedBytes;
            internal long TotalBytes;
            internal DateTime LastActivityUtc = DateTime.UtcNow;
            internal bool NetworkPhase = true;
            internal bool Installing;
            internal int TimeoutRemaining = 120;

            internal PipDisplayState(string stage, string source, int sourceIndex, int sourceCount)
            {
                Stage = stage;
                Source = source;
                SourceIndex = sourceIndex;
                SourceCount = sourceCount;
            }
        }

        private sealed class ProcessResult
        {
            internal readonly int ExitCode;
            internal readonly string Output;
            internal readonly string Error;
            internal readonly bool TimedOut;

            internal ProcessResult(int exitCode, string output, string error, bool timedOut = false)
            {
                ExitCode = exitCode;
                Output = output ?? string.Empty;
                Error = error ?? string.Empty;
                TimedOut = timedOut;
            }
        }
    }
}


