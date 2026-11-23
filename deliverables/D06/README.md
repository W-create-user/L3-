# D06 接管与安全恢复验收

在仓库根目录运行：

```powershell
.\.venv\Scripts\python -m l3_system.monitoring.takeover_acceptance --real --output outputs/d06_new_run
.\.venv\Scripts\python -m unittest discover -s deliverables/D06/tests -v
```

输出目录必须不存在。`--real` 实际创建 SMPL Atropine；不传时只运行7个契约场景。真实场景为正常、可恢复拒绝和 episode 终止。终止时按真实 done 标志检查，不假设 max_steps=1 就只调用一次 step。

7个契约场景覆盖正常、空确认、未释放 C 资源、未结 B 请求、缺失关键观测、过期审批、未知执行状态。多个场景同时复核未确认阻断、缺少表单凭据、重复表单、重复确认、确认零step以及恢复后的动作仍经 D 审核。测试只在夹具中构造锁和缺失观测，真实模型不作这类内部篡改。

每场景保存 B 记录、D 审核、系统转移、HTTP状态、UTC与单调时间、配置/源码指纹和结果。`summary.json` 是本地模拟接管验证；操作人由 HTTP 测试客户端模拟，未验证浏览器排版，未招募志愿者，不代表人因研究。D05 独立风险建议器和完整三模态模型仍未接入 A 的 Web 会话，不能由本组结果推断已集成。

原始仿真轨迹保留在指定输出目录；提交公共代码库时仅提交审核过的汇总与指纹。
