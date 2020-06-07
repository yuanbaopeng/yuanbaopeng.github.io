---
layout: post
title:  "Tensor Plugin使用"
date:   2020-06-07 20:03:29 +0800
---

```c++
nvinfer1::IPluginV2* importPluginFromRegistry(IImporterContext* ctx, const std::string& pluginName,
    const std::string& pluginVersion, const std::string& nodeName,
    const std::vector<nvinfer1::PluginField>& pluginFields)
{
    const auto mPluginRegistry = getPluginRegistry();
    const auto pluginCreator
        = mPluginRegistry->getPluginCreator(pluginName.c_str(), pluginVersion.c_str(), "ONNXTRT_NAMESPACE");

    if (!pluginCreator)
    {
        return nullptr;
    }

    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = pluginFields.size();
    fc.fields = pluginFields.data();

    return pluginCreator->createPlugin(nodeName.c_str(), &fc);
}
```

```c++
// Create plugin from registry
nvinfer1::IPluginV2* plugin = importPluginFromRegistry(ctx, pluginName, pluginVersion, node.name(), f);

ASSERT(plugin != nullptr && "InstanceNormalization plugin was not found in the plugin registry!",
    ErrorCode::kUNSUPPORTED_NODE);

RETURN_FIRST_OUTPUT(ctx->network()->addPluginV2(&tensor_ptr, 1, *plugin));
```